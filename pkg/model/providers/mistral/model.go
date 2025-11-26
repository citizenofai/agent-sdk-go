package mistral

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
	"unicode"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"

	"github.com/citizenofai/agent-sdk-go/pkg/model"
	mistralsdk "github.com/gage-technologies/mistral-go"
)

// Model implements the model.Model interface for Mistral using the mistral-go SDK.
type Model struct {
	ModelName string
	Provider  *Provider
}

// chatCompletionResponse mirrors the subset of the Mistral chat completion response
// we care about, with Content as json.RawMessage so we can handle both string and
// array-of-parts formats without JSON unmarshal errors.
type chatCompletionResponse struct {
	ID      string                  `json:"id"`
	Object  string                  `json:"object"`
	Created int64                   `json:"created"`
	Model   string                  `json:"model"`
	Choices []chatCompletionChoice  `json:"choices"`
	Usage   chatCompletionUsageInfo `json:"usage"`
}

type chatCompletionChoice struct {
	Index        int                   `json:"index"`
	Message      chatCompletionMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
}

type chatCompletionMessage struct {
	Role      string                   `json:"role"`
	Content   json.RawMessage          `json:"content"`
	ToolCalls []chatCompletionToolCall `json:"tool_calls,omitempty"`
}

type chatCompletionToolCall struct {
	ID       string                         `json:"id"`
	Type     string                         `json:"type"`
	Function chatCompletionToolCallFunction `json:"function"`
}

type chatCompletionToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatCompletionUsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// GetResponse gets a single response from the model with retry logic.
func (m *Model) GetResponse(ctx context.Context, request *model.Request) (*model.Response, error) {
	var (
		resp    *model.Response
		lastErr error
	)

	for attempt := 0; attempt <= m.Provider.MaxRetries; attempt++ {
		m.Provider.WaitForRateLimit()

		if attempt > 0 {
			backoff := calculateBackoff(attempt, m.Provider.RetryAfter)
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("mistral: context cancelled during backoff: %w", ctx.Err())
			case <-time.After(backoff):
			}
		}

		resp, lastErr = m.getResponseOnce(ctx, request)
		if lastErr == nil {
			return resp, nil
		}
		if !isRateLimitError(lastErr) {
			return nil, lastErr
		}
	}

	return nil, lastErr
}

// getResponseOnce sends a single Chat call via mistral-go.
func (m *Model) getResponseOnce(ctx context.Context, request *model.Request) (*model.Response, error) {
	messages := buildChatMessagesFromRequest(request)
	if len(messages) == 0 {
		return nil, fmt.Errorf("mistral: empty request input")
	}

	params := buildChatParamsFromRequest(request)

	// Build request payload following mistral-go's Chat implementation to ensure
	// compatibility with tool_choice and response_format expectations.
	requestData := map[string]interface{}{
		"model":       m.ModelName,
		"messages":    messages,
		"temperature": params.Temperature,
		"max_tokens":  params.MaxTokens,
		"top_p":       params.TopP,
		"random_seed": params.RandomSeed,
		"safe_prompt": params.SafePrompt,
	}
	if params.Tools != nil {
		requestData["tools"] = params.Tools
	}
	if params.ToolChoice != "" {
		requestData["tool_choice"] = params.ToolChoice
	}
	if params.ResponseFormat != "" {
		requestData["response_format"] = map[string]any{"type": params.ResponseFormat}
	}

	// Determine endpoint
	endpoint := m.Provider.endpoint
	if endpoint == "" {
		endpoint = mistralsdk.Endpoint
	}
	url := strings.TrimRight(endpoint, "/") + "/v1/chat/completions"

	// Marshal request body
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(requestData); err != nil {
		return nil, fmt.Errorf("mistral: failed to encode request: %w", err)
	}

	// Execute HTTP request
	httpClient := m.Provider.getHTTPClient()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &buf)
	if err != nil {
		return nil, fmt.Errorf("mistral: failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.Provider.APIKey)

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("mistral: HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("mistral: API error %d: %s", resp.StatusCode, strings.TrimSpace(string(bodyBytes)))
	}

	var apiResp chatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("mistral: failed to decode response: %w", err)
	}

	if len(apiResp.Choices) == 0 {
		return nil, fmt.Errorf("mistral: no choices in response")
	}

	choice := apiResp.Choices[0]

	usage := &model.Usage{
		PromptTokens:     apiResp.Usage.PromptTokens,
		CompletionTokens: apiResp.Usage.CompletionTokens,
		TotalTokens:      apiResp.Usage.TotalTokens,
	}
	if usage.TotalTokens > 0 {
		m.Provider.UpdateTokenCount(usage.TotalTokens)
	}

	toolCalls := make([]model.ToolCall, 0, len(choice.Message.ToolCalls))
	for _, tc := range choice.Message.ToolCalls {
		toolCall := model.ToolCall{
			ID:           tc.ID,
			Name:         tc.Function.Name,
			Parameters:   map[string]interface{}{},
			RawParameter: strings.Builder{},
		}
		if tc.Function.Arguments != "" {
			toolCall.RawParameter.WriteString(tc.Function.Arguments)
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err == nil {
				for k, v := range args {
					toolCall.Parameters[k] = v
				}
			}
		}
		toolCalls = append(toolCalls, toolCall)
	}

	handoffCall, handoffIdx := detectHandoffFromToolCalls(toolCalls)
	if handoffIdx >= 0 {
		toolCalls = removeToolCallAt(toolCalls, handoffIdx)
	}

	return &model.Response{
		Content:     normalizeMistralContent(choice.Message.Content),
		ToolCalls:   toolCalls,
		HandoffCall: handoffCall,
		Usage:       usage,
	}, nil
}

// StreamResponse streams a response from the model with retry logic.
func (m *Model) StreamResponse(ctx context.Context, request *model.Request) (<-chan model.StreamEvent, error) {
	events := make(chan model.StreamEvent)

	go func() {
		defer close(events)

		var lastErr error

		for attempt := 0; attempt <= m.Provider.MaxRetries; attempt++ {
			m.Provider.WaitForRateLimit()

			if attempt > 0 {
				backoff := calculateBackoff(attempt, m.Provider.RetryAfter)
				select {
				case <-ctx.Done():
					events <- model.StreamEvent{Type: model.StreamEventTypeError, Error: fmt.Errorf("mistral: context cancelled during backoff: %w", ctx.Err())}
					return
				case <-time.After(backoff):
				}
			}

			if err := m.streamResponseOnce(ctx, request, events); err != nil {
				lastErr = err
				if !isRateLimitError(err) || ctx.Err() != nil {
					if ctx.Err() != nil {
						err = fmt.Errorf("mistral: context cancelled: %w", ctx.Err())
					}
					events <- model.StreamEvent{Type: model.StreamEventTypeError, Error: err}
					return
				}
				continue
			}
			return
		}

		if lastErr != nil {
			events <- model.StreamEvent{Type: model.StreamEventTypeError, Error: lastErr}
		}
	}()

	return events, nil
}

// streamResponseOnce performs a single streaming call and emits events.
func (m *Model) streamResponseOnce(_ context.Context, request *model.Request, events chan<- model.StreamEvent) error {
	client := m.Provider.getClient()

	messages := buildChatMessagesFromRequest(request)
	if len(messages) == 0 {
		return fmt.Errorf("mistral: empty request input")
	}

	params := buildChatParamsFromRequest(request)

	stream, err := client.ChatStream(m.ModelName, messages, params)
	if err != nil {
		return err
	}

	var (
		contentBuilder strings.Builder
		toolCalls      []model.ToolCall
		usage          *model.Usage
	)

	for chunk := range stream {
		if chunk.Error != nil {
			return chunk.Error
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]

		if chunk.Usage.TotalTokens > 0 {
			usage = &model.Usage{
				PromptTokens:     chunk.Usage.PromptTokens,
				CompletionTokens: chunk.Usage.CompletionTokens,
				TotalTokens:      chunk.Usage.TotalTokens,
			}
		}

		if choice.Delta.Content != "" {
			contentBuilder.WriteString(choice.Delta.Content)
			events <- model.StreamEvent{Type: model.StreamEventTypeContent, Content: choice.Delta.Content}
		}

		for _, tc := range choice.Delta.ToolCalls {
			toolCall := model.ToolCall{
				ID:           tc.Id,
				Name:         tc.Function.Name,
				Parameters:   map[string]interface{}{},
				RawParameter: strings.Builder{},
			}
			if tc.Function.Arguments != "" {
				toolCall.RawParameter.WriteString(tc.Function.Arguments)
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err == nil {
					for k, v := range args {
						toolCall.Parameters[k] = v
					}
				}
			}
			toolCalls = append(toolCalls, toolCall)
			last := toolCalls[len(toolCalls)-1]
			events <- model.StreamEvent{Type: model.StreamEventTypeToolCall, ToolCall: &last}
		}
	}

	handoffCall, handoffIdx := detectHandoffFromToolCalls(toolCalls)
	if handoffIdx >= 0 {
		toolCalls = removeToolCallAt(toolCalls, handoffIdx)
	}

	if usage != nil && usage.TotalTokens > 0 {
		m.Provider.UpdateTokenCount(usage.TotalTokens)
	}

	events <- model.StreamEvent{
		Type: model.StreamEventTypeDone,
		Response: &model.Response{
			Content:     contentBuilder.String(),
			ToolCalls:   toolCalls,
			HandoffCall: handoffCall,
			Usage:       usage,
		},
	}

	return nil
}

// buildChatMessagesFromRequest flattens a model.Request into Mistral chat messages.
func buildChatMessagesFromRequest(req *model.Request) []mistralsdk.ChatMessage {
	if req == nil {
		return nil
	}

	var messages []mistralsdk.ChatMessage

	if strings.TrimSpace(req.SystemInstructions) != "" {
		messages = append(messages, mistralsdk.ChatMessage{
			Role:    mistralsdk.RoleSystem,
			Content: req.SystemInstructions,
		})
	}

	switch v := req.Input.(type) {
	case nil:
		// no-op
	case string:
		if strings.TrimSpace(v) != "" {
			messages = append(messages, mistralsdk.ChatMessage{
				Role:    mistralsdk.RoleUser,
				Content: v,
			})
		}
	case []interface{}:
		for _, item := range v {
			msg, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			role, _ := msg["role"].(string)
			if role == "" {
				role = string(mistralsdk.RoleUser)
			}
			content, _ := msg["content"].(string)
			if strings.TrimSpace(content) == "" {
				continue
			}
			messages = append(messages, mistralsdk.ChatMessage{
				Role:    role,
				Content: content,
			})
		}
	default:
		messages = append(messages, mistralsdk.ChatMessage{
			Role:    mistralsdk.RoleUser,
			Content: fmt.Sprintf("%v", v),
		})
	}

	return messages
}

// buildChatParamsFromRequest builds ChatRequestParams from a generic request.
func buildChatParamsFromRequest(req *model.Request) *mistralsdk.ChatRequestParams {
	params := mistralsdk.DefaultChatRequestParams

	if req != nil && req.Settings != nil {
		if req.Settings.Temperature != nil {
			params.Temperature = *req.Settings.Temperature
		}
		if req.Settings.TopP != nil {
			params.TopP = *req.Settings.TopP
		}
		if req.Settings.MaxTokens != nil {
			params.MaxTokens = *req.Settings.MaxTokens
		}
		if req.Settings.ToolChoice != nil {
			choice := strings.ToLower(*req.Settings.ToolChoice)
			switch choice {
			case "none":
				params.ToolChoice = mistralsdk.ToolChoiceNone
			case "auto":
				params.ToolChoice = mistralsdk.ToolChoiceAuto
			case "any":
				params.ToolChoice = mistralsdk.ToolChoiceAny
			}
		}
	}

	params.Tools = buildToolsFromRequest(req)

	if req != nil && req.OutputSchema != nil {
		params.ResponseFormat = mistralsdk.ResponseFormatJsonObject
	}

	return &params
}

// buildToolsFromRequest converts Request.Tools and Request.Handoffs into mistral Tools.
func buildToolsFromRequest(req *model.Request) []mistralsdk.Tool {
	if req == nil {
		return nil
	}

	var tools []mistralsdk.Tool

	for _, t := range req.Tools {
		if tool := convertToMistralTool(t); tool != nil {
			tools = append(tools, *tool)
		}
	}
	for _, h := range req.Handoffs {
		if tool := convertToMistralTool(h); tool != nil {
			tools = append(tools, *tool)
		}
	}

	if len(tools) == 0 {
		return nil
	}

	return tools
}

// convertToMistralTool converts a generic tool/handoff definition into a mistral Tool.
func convertToMistralTool(tool interface{}) *mistralsdk.Tool {
	if tool == nil {
		return nil
	}

	var name string
	var description string
	var params any

	if m, ok := tool.(map[string]interface{}); ok {
		if m["type"] == "function" && m["function"] != nil {
			if fn, ok := m["function"].(map[string]interface{}); ok {
				if v, ok := fn["name"].(string); ok {
					name = v
				}
				if v, ok := fn["description"].(string); ok {
					description = v
				}
				if p, ok := fn["parameters"]; ok {
					params = p
				}
			}
		} else if m["name"] != nil {
			if v, ok := m["name"].(string); ok {
				name = v
			}
			if v, ok := m["description"].(string); ok {
				description = v
			}
			if p, ok := m["parameters"]; ok {
				params = p
			}
		} else {
			return nil
		}
	} else {
		if ti, ok := tool.(interface {
			GetName() string
			GetDescription() string
			GetParametersSchema() map[string]interface{}
		}); ok {
			name = ti.GetName()
			description = ti.GetDescription()
			params = ti.GetParametersSchema()
		} else if ti, ok := tool.(interface {
			GetName() string
			GetDescription() string
		}); ok {
			name = ti.GetName()
			description = ti.GetDescription()
		}
	}

	name = sanitizeFunctionName(name)
	if name == "" {
		return nil
	}

	return &mistralsdk.Tool{
		Type: mistralsdk.ToolTypeFunction,
		Function: mistralsdk.Function{
			Name:        name,
			Description: description,
			Parameters:  params,
		},
	}
}

// sanitizeFunctionName normalizes a function/tool name for Mistral.
func sanitizeFunctionName(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return ""
	}

	var b strings.Builder
	for i, r := range name {
		allowed := unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '.' || r == ':' || r == '-'
		if !allowed {
			r = '_'
		}
		if i == 0 {
			if !(unicode.IsLetter(r) || r == '_') {
				b.WriteRune('f')
				b.WriteRune('_')
			}
		}
		b.WriteRune(r)
	}

	res := b.String()
	if len(res) > 64 {
		res = res[:64]
	}
	return res
}

// detectHandoffFromToolCalls inspects tool calls and extracts a HandoffCall if present.
func detectHandoffFromToolCalls(toolCalls []model.ToolCall) (*model.HandoffCall, int) {
	for idx, tc := range toolCalls {
		nameLower := strings.ToLower(tc.Name)
		args := tc.Parameters

		if strings.HasPrefix(nameLower, "handoff_to_") {
			agentName := strings.TrimPrefix(tc.Name, "handoff_to_")
			input, _ := getStringArg(args, "input")
			handoff := &model.HandoffCall{
				AgentName:      agentName,
				Parameters:     map[string]interface{}{"input": input},
				Type:           model.HandoffTypeDelegate,
				ReturnToAgent:  "",
				TaskID:         "",
				IsTaskComplete: false,
			}
			if taskID, ok := getStringArg(args, "task_id"); ok && taskID != "" {
				handoff.TaskID = taskID
			}
			if returnTo, ok := getStringArg(args, "return_to_agent"); ok && returnTo != "" {
				handoff.ReturnToAgent = returnTo
			}
			if isComplete, ok := getBoolArg(args, "is_task_complete"); ok {
				handoff.IsTaskComplete = isComplete
			}
			return handoff, idx
		}

		if strings.HasPrefix(nameLower, "handoff") {
			agentName, _ := getStringArg(args, "agent")
			if agentName != "" {
				input, _ := getStringArg(args, "input")
				if input == "" {
					inputMap := make(map[string]interface{})
					for k, v := range args {
						if k != "agent" && k != "task_id" && k != "return_to_agent" && k != "is_task_complete" {
							inputMap[k] = v
						}
					}
					if raw, err := json.Marshal(inputMap); err == nil {
						input = string(raw)
					}
				}

				handoff := &model.HandoffCall{
					AgentName:      agentName,
					Parameters:     map[string]interface{}{"input": input},
					Type:           model.HandoffTypeDelegate,
					ReturnToAgent:  "",
					TaskID:         "",
					IsTaskComplete: false,
				}
				if taskID, ok := getStringArg(args, "task_id"); ok && taskID != "" {
					handoff.TaskID = taskID
				}
				if returnTo, ok := getStringArg(args, "return_to_agent"); ok && returnTo != "" {
					handoff.ReturnToAgent = returnTo
				}
				if isComplete, ok := getBoolArg(args, "is_task_complete"); ok {
					handoff.IsTaskComplete = isComplete
				}
				if agentName == "return_to_delegator" || strings.EqualFold(agentName, "return") {
					handoff.Type = model.HandoffTypeReturn
				}
				return handoff, idx
			}
		}

		if strings.Contains(nameLower, "agent") {
			possibleAgentName := strings.ReplaceAll(nameLower, "_agent", " agent")
			possibleAgentName = cases.Title(language.Und, cases.NoLower).String(possibleAgentName)
			if strings.HasSuffix(possibleAgentName, "Agent") {
				handoff := &model.HandoffCall{
					AgentName:      possibleAgentName,
					Parameters:     args,
					Type:           model.HandoffTypeDelegate,
					ReturnToAgent:  "",
					TaskID:         "",
					IsTaskComplete: false,
				}
				if taskID, ok := getStringArg(args, "task_id"); ok && taskID != "" {
					handoff.TaskID = taskID
				}
				if returnTo, ok := getStringArg(args, "return_to_agent"); ok && returnTo != "" {
					handoff.ReturnToAgent = returnTo
				}
				if isComplete, ok := getBoolArg(args, "is_task_complete"); ok {
					handoff.IsTaskComplete = isComplete
				}
				return handoff, idx
			}
		}
	}

	return nil, -1
}

// removeToolCallAt removes a tool call at the given index.
func removeToolCallAt(calls []model.ToolCall, idx int) []model.ToolCall {
	if idx < 0 || idx >= len(calls) {
		return calls
	}
	return append(calls[:idx], calls[idx+1:]...)
}

func getStringArg(args map[string]interface{}, key string) (string, bool) {
	if args == nil {
		return "", false
	}
	if v, ok := args[key]; ok {
		if s, ok := v.(string); ok {
			return s, true
		}
	}
	return "", false
}

func getBoolArg(args map[string]interface{}, key string) (bool, bool) {
	if args == nil {
		return false, false
	}
	if v, ok := args[key]; ok {
		if b, ok := v.(bool); ok {
			return b, true
		}
	}
	return false, false
}

// isRateLimitError checks if an error is likely a rate limit error.
func isRateLimitError(err error) bool {
	if err == nil {
		return false
	}
	var apiErr interface{ Error() string }
	if errors.As(err, &apiErr) {
		s := apiErr.Error()
		return strings.Contains(s, "429") || strings.Contains(strings.ToLower(s), "rate limit")
	}
	s := err.Error()
	return strings.Contains(s, "429") || strings.Contains(strings.ToLower(s), "rate limit")
}

// normalizeMistralContent converts the raw content field from a Mistral chat message
// into a plain string. Mistral may return either a simple string or an array of
// content parts; this function safely handles both without JSON unmarshal errors.
func normalizeMistralContent(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}

	// First, try simple string content.
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}

	// Next, try an array of content parts with optional "text" fields.
	var parts []map[string]interface{}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var b strings.Builder
		for _, p := range parts {
			if txt, ok := p["text"].(string); ok {
				b.WriteString(txt)
			}
		}
		if b.Len() > 0 {
			return b.String()
		}
	}

	// Fallback: return the raw JSON as a string so we don't lose information.
	return string(raw)
}

// calculateBackoff calculates the backoff duration for retries with jitter.
func calculateBackoff(attempt int, baseDelay time.Duration) time.Duration {
	if baseDelay <= 0 {
		baseDelay = time.Second
	}
	backoff := float64(baseDelay) * math.Pow(2, float64(attempt))
	b := make([]byte, 1)
	if _, err := rand.Read(b); err != nil {
		return time.Duration(backoff)
	}
	jitter := float64(b[0]) / 255.0 * (backoff / 2)
	return time.Duration(backoff + jitter)
}
