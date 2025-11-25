package gemini

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"
	"unicode"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"
	genai "google.golang.org/genai"

	"github.com/citizenofai/agent-sdk-go/pkg/model"
)

// handoffFunctionNameToAgentName maps sanitized Gemini function names for
// handoff tools (e.g. "handoff_to_Math_Agent") back to the original agent
// names as configured on the Agent (e.g. "Math Agent"). This lets us keep
// function names Gemini-compliant without losing the true agent identifiers
// used by the runner.
var handoffFunctionNameToAgentName = map[string]string{}

// Model implements the model.Model interface for Gemini using the go-genai SDK.
type Model struct {
	ModelName string
	Provider  *Provider
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
				return nil, fmt.Errorf("gemini: context cancelled during backoff: %w", ctx.Err())
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

// getResponseOnce sends a single GenerateContent call via go-genai.
func (m *Model) getResponseOnce(ctx context.Context, request *model.Request) (*model.Response, error) {
	client, err := m.Provider.getClient(ctx)
	if err != nil {
		return nil, err
	}

	text := buildInputText(request)
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("gemini: empty request input")
	}

	contents := []*genai.Content{{
		Parts: []*genai.Part{{Text: text}},
	}}
	config := buildGenerateContentConfig(request)

	apiResp, err := client.Models.GenerateContent(ctx, m.ModelName, contents, config)
	if err != nil {
		return nil, err
	}

	if len(apiResp.Candidates) == 0 || apiResp.Candidates[0].Content == nil {
		return nil, fmt.Errorf("gemini: no candidates in response")
	}

	var (
		contentBuilder strings.Builder
		toolCalls      []model.ToolCall
		usage          *model.Usage
	)

	// Map usage metadata if available
	if apiResp.UsageMetadata != nil {
		u := apiResp.UsageMetadata
		usage = &model.Usage{
			PromptTokens:     int(u.PromptTokenCount),
			CompletionTokens: int(u.CandidatesTokenCount),
			TotalTokens:      int(u.TotalTokenCount),
		}
		if usage.TotalTokens > 0 {
			m.Provider.UpdateTokenCount(usage.TotalTokens)
		}
	}

	for _, candidate := range apiResp.Candidates {
		if candidate == nil || candidate.Content == nil {
			continue
		}
		for _, part := range candidate.Content.Parts {
			if part == nil {
				continue
			}
			if part.Text != "" {
				contentBuilder.WriteString(part.Text)
			}
			if fc := part.FunctionCall; fc != nil {
				toolCall := model.ToolCall{
					Name:         fc.Name,
					Parameters:   map[string]interface{}{},
					RawParameter: strings.Builder{},
				}
				if fc.Args != nil {
					for k, v := range fc.Args {
						toolCall.Parameters[k] = v
					}
					if raw, err := json.Marshal(fc.Args); err == nil {
						toolCall.RawParameter.Write(raw)
					}
				}
				toolCalls = append(toolCalls, toolCall)
			}
		}
	}

	handoffCall, handoffIdx := detectHandoffFromToolCalls(toolCalls)
	if handoffIdx >= 0 {
		toolCalls = removeToolCallAt(toolCalls, handoffIdx)
	}

	// UsageMetadata may not be populated by all endpoints; we ignore detailed tokens here.
	return &model.Response{
		Content:     contentBuilder.String(),
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
					events <- model.StreamEvent{Type: model.StreamEventTypeError, Error: fmt.Errorf("gemini: context cancelled during backoff: %w", ctx.Err())}
					return
				case <-time.After(backoff):
				}
			}

			if err := m.streamResponseOnce(ctx, request, events); err != nil {
				lastErr = err
				if !isRateLimitError(err) || ctx.Err() != nil {
					if ctx.Err() != nil {
						err = fmt.Errorf("gemini: context cancelled: %w", ctx.Err())
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
func (m *Model) streamResponseOnce(ctx context.Context, request *model.Request, events chan<- model.StreamEvent) error {
	client, err := m.Provider.getClient(ctx)
	if err != nil {
		return err
	}

	text := buildInputText(request)
	if strings.TrimSpace(text) == "" {
		return fmt.Errorf("gemini: empty request input")
	}

	contents := []*genai.Content{{
		Parts: []*genai.Part{{Text: text}},
	}}
	config := buildGenerateContentConfig(request)

	var (
		content   strings.Builder
		toolCalls []model.ToolCall
		usage     *model.Usage
	)

	for apiResp, err := range client.Models.GenerateContentStream(ctx, m.ModelName, contents, config) {
		if err != nil {
			return err
		}
		if apiResp.UsageMetadata != nil {
			u := apiResp.UsageMetadata
			usage = &model.Usage{
				PromptTokens:     int(u.PromptTokenCount),
				CompletionTokens: int(u.CandidatesTokenCount),
				TotalTokens:      int(u.TotalTokenCount),
			}
		}

		if len(apiResp.Candidates) == 0 || apiResp.Candidates[0].Content == nil {
			continue
		}
		for _, part := range apiResp.Candidates[0].Content.Parts {
			if part == nil {
				continue
			}
			if part.Text != "" {
				content.WriteString(part.Text)
				events <- model.StreamEvent{Type: model.StreamEventTypeContent, Content: part.Text}
			}
			if fc := part.FunctionCall; fc != nil {
				toolCall := model.ToolCall{
					Name:         fc.Name,
					Parameters:   map[string]interface{}{},
					RawParameter: strings.Builder{},
				}
				if fc.Args != nil {
					for k, v := range fc.Args {
						toolCall.Parameters[k] = v
					}
					if raw, err := json.Marshal(fc.Args); err == nil {
						toolCall.RawParameter.Write(raw)
					}
				}
				toolCalls = append(toolCalls, toolCall)
				// Emit tool call event as it arrives
				last := toolCalls[len(toolCalls)-1]
				events <- model.StreamEvent{Type: model.StreamEventTypeToolCall, ToolCall: &last}
			}
		}
	}

	handoffCall, handoffIdx := detectHandoffFromToolCalls(toolCalls)
	if handoffIdx >= 0 {
		toolCalls = removeToolCallAt(toolCalls, handoffIdx)
	}
	if handoffCall != nil {
		events <- model.StreamEvent{Type: model.StreamEventTypeHandoff, HandoffCall: handoffCall}
	}
	if usage != nil && usage.TotalTokens > 0 {
		m.Provider.UpdateTokenCount(usage.TotalTokens)
	}

	events <- model.StreamEvent{
		Type: model.StreamEventTypeDone,
		Response: &model.Response{
			Content:     content.String(),
			ToolCalls:   toolCalls,
			HandoffCall: handoffCall,
			Usage:       usage,
		},
	}

	return nil
}

// buildInputText flattens a model.Request into a single text prompt.
func buildInputText(req *model.Request) string {
	var sb strings.Builder
	if req == nil {
		return ""
	}
	if strings.TrimSpace(req.SystemInstructions) != "" {
		sb.WriteString(req.SystemInstructions)
		sb.WriteString("\n\n")
	}
	switch v := req.Input.(type) {
	case nil:
		// no-op
	case string:
		sb.WriteString(v)
	case []interface{}:
		for _, item := range v {
			if msg, ok := item.(map[string]interface{}); ok {
				if c, ok := msg["content"].(string); ok {
					if sb.Len() > 0 {
						sb.WriteString("\n")
					}
					sb.WriteString(c)
				}
			}
		}
	default:
		sb.WriteString(fmt.Sprintf("%v", v))
	}
	return sb.String()
}

// TestBuildInputText is a small helper to expose buildInputText for tests.
func TestBuildInputText(req *model.Request) string {
	return buildInputText(req)
}

// buildGenerateContentConfig builds a genai.GenerateContentConfig from a generic request.
// It currently maps tools/handoffs and a subset of model.Settings.
func buildGenerateContentConfig(req *model.Request) *genai.GenerateContentConfig {
	if req == nil {
		return nil
	}

	cfg := &genai.GenerateContentConfig{}

	// Tools and handoffs -> function declarations
	tools := buildToolsFromRequest(req)
	if len(tools) > 0 {
		// Map ToolChoice semantics as closely as possible:
		// - "none": disable tools entirely
		// - "auto" or specific tool name: we still expose all tools; Gemini's
		//   current Go SDK does not expose a per-function selection field.
		if req.Settings != nil && req.Settings.ToolChoice != nil {
			choice := strings.ToLower(*req.Settings.ToolChoice)
			if choice == "none" {
				tools = nil
			}
		}
		if len(tools) > 0 {
			cfg.Tools = tools
		}
	}

	// Basic generation settings
	if req.Settings != nil {
		if req.Settings.Temperature != nil {
			cfg.Temperature = genai.Ptr[float32](float32(*req.Settings.Temperature))
		}
		if req.Settings.TopP != nil {
			cfg.TopP = genai.Ptr[float32](float32(*req.Settings.TopP))
		}
		if req.Settings.MaxTokens != nil {
			cfg.MaxOutputTokens = int32(*req.Settings.MaxTokens)
		}
	}

	// If there are no tools and no settings, we can return nil to avoid
	// sending an entirely empty config. Otherwise, return cfg.
	if len(cfg.Tools) == 0 && req.Settings == nil {
		return nil
	}

	return cfg
}

// buildToolsFromRequest converts Request.Tools and Request.Handoffs into genai.Tools.
func buildToolsFromRequest(req *model.Request) []*genai.Tool {
	var decls []*genai.FunctionDeclaration

	for _, t := range req.Tools {
		if fd := convertToolToFunctionDecl(t); fd != nil {
			decls = append(decls, fd)
		}
	}
	for _, h := range req.Handoffs {
		if fd := convertToolToFunctionDecl(h); fd != nil {
			decls = append(decls, fd)
		}
	}

	if len(decls) == 0 {
		return nil
	}

	// Group all function declarations into a single Tool, which is sufficient for Gemini.
	return []*genai.Tool{{FunctionDeclarations: decls}}
}

// convertToolToFunctionDecl converts a generic tool/handoff definition into a FunctionDeclaration.
// It supports both OpenAI-style maps and simple tool interfaces.
func convertToolToFunctionDecl(tool interface{}) *genai.FunctionDeclaration {
	if tool == nil {
		return nil
	}

	var name string
	var description string
	var params map[string]interface{}

	if m, ok := tool.(map[string]interface{}); ok {
		// OpenAI-style: {"type":"function","function":{...}}
		if m["type"] == "function" && m["function"] != nil {
			if fn, ok := m["function"].(map[string]interface{}); ok {
				if v, ok := fn["name"].(string); ok {
					name = v
				}
				if v, ok := fn["description"].(string); ok {
					description = v
				}
				if p, ok := fn["parameters"].(map[string]interface{}); ok {
					params = p
				}
			}
		} else if m["name"] != nil {
			// Legacy simple format: {"name":..., "description":..., "parameters":...}
			if v, ok := m["name"].(string); ok {
				name = v
			}
			if v, ok := m["description"].(string); ok {
				description = v
			}
			if p, ok := m["parameters"].(map[string]interface{}); ok {
				params = p
			}
		} else {
			return nil
		}
	} else {
		// Tool interface with basic metadata (+ optional schema)
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
		} else {
			return nil
		}
	}

	// Remember original name for handoff tools so we can recover the real
	// agent name even after sanitization.
	originalName := name

	// Sanitize to comply with Gemini function name requirements.
	name = sanitizeFunctionName(name)
	if name == "" {
		return nil
	}

	var schema *genai.Schema
	if params != nil {
		schema = schemaFromJSONSchema(params)
	}
	if schema == nil {
		schema = &genai.Schema{Type: genai.TypeObject}
	}

	// If this is a handoff tool (name starts with handoff_to_), record the
	// mapping from the sanitized function name back to the original agent
	// name suffix from the definition. This allows HandoffCall.AgentName to
	// remain equal to the underlying Agent.Name even though Gemini sees the
	// sanitized function name.
	if strings.HasPrefix(strings.ToLower(originalName), "handoff_to_") {
		agentName := strings.TrimPrefix(originalName, "handoff_to_")
		if agentName != "" {
			handoffFunctionNameToAgentName[name] = agentName
		}
	}

	return &genai.FunctionDeclaration{
		Name:        name,
		Description: description,
		Parameters:  schema,
	}
}

// sanitizeFunctionName normalizes a function/tool name to comply with
// Gemini's function naming rules:
//   - Must start with a letter or underscore
//   - Subsequent chars may be letters, digits, '_', '.', ':', or '-'
//   - Max length 64
//
// We also preserve patterns like "handoff_to_" and "agent" by only
// replacing disallowed characters (e.g. spaces) with underscores.
func sanitizeFunctionName(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return ""
	}

	var b strings.Builder
	for i, r := range name {
		allowed := unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '.' || r == ':' || r == '-'
		if !allowed {
			// Replace disallowed characters (e.g. spaces) with underscore
			r = '_'
		}
		if i == 0 {
			// First char must be letter or underscore
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

// schemaFromJSONSchema converts a JSON Schema-like map (as used in tool parameters)
// into a genai.Schema tree. It supports the common subset used in this SDK
// (type, properties, required, description, enum).
func schemaFromJSONSchema(m map[string]interface{}) *genai.Schema {
	s := &genai.Schema{Type: genai.TypeObject}
	if m == nil {
		return s
	}

	if desc, ok := m["description"].(string); ok {
		s.Description = desc
	}
	if enumRaw, ok := m["enum"].([]interface{}); ok {
		enums := make([]string, 0, len(enumRaw))
		for _, v := range enumRaw {
			if str, ok := v.(string); ok {
				enums = append(enums, str)
			}
		}
		s.Enum = enums
	}

	if t, ok := m["type"].(string); ok {
		switch strings.ToLower(t) {
		case "string":
			s.Type = genai.TypeString
		case "integer":
			s.Type = genai.TypeInteger
		case "number":
			s.Type = genai.TypeNumber
		case "boolean":
			s.Type = genai.TypeBoolean
		case "array":
			s.Type = genai.TypeArray
		case "object":
			fallthrough
		default:
			s.Type = genai.TypeObject
		}
	}

	// Nested object properties
	if props, ok := m["properties"].(map[string]interface{}); ok {
		s.Properties = make(map[string]*genai.Schema, len(props))
		for name, raw := range props {
			if pm, ok := raw.(map[string]interface{}); ok {
				s.Properties[name] = schemaFromJSONSchema(pm)
			}
		}
	}

	// Required fields
	if reqRaw, ok := m["required"].([]interface{}); ok {
		req := make([]string, 0, len(reqRaw))
		for _, v := range reqRaw {
			if str, ok := v.(string); ok {
				req = append(req, str)
			}
		}
		s.Required = req
	}

	return s
}

// detectHandoffFromToolCalls inspects tool calls and extracts a HandoffCall if present.
// It follows the same naming conventions as the OpenAI provider (handoff_to_*, handoff*, *agent).
func detectHandoffFromToolCalls(toolCalls []model.ToolCall) (*model.HandoffCall, int) {
	for idx, tc := range toolCalls {
		nameLower := strings.ToLower(tc.Name)
		args := tc.Parameters

		// Pattern 1: handoff_to_<agent>
		if strings.HasPrefix(nameLower, "handoff_to_") {
			agentName := strings.TrimPrefix(tc.Name, "handoff_to_")
			if mapped, ok := handoffFunctionNameToAgentName[tc.Name]; ok && mapped != "" {
				agentName = mapped
			}
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

		// Pattern 2: generic "handoff" tool with explicit agent in args
		if strings.HasPrefix(nameLower, "handoff") {
			agentName, _ := getStringArg(args, "agent")
			if agentName != "" {
				input, _ := getStringArg(args, "input")
				if input == "" {
					// Derive input from remaining args
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

				// Detect explicit return handoff
				if agentName == "return_to_delegator" || strings.EqualFold(agentName, "return") {
					handoff.Type = model.HandoffTypeReturn
				}

				return handoff, idx
			}
		}

		// Pattern 3: tool name that looks like an agent (e.g. "support_agent")
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
	// If this is a genai.APIError, inspect its status/code.
	var apiErr interface{ Error() string }
	if errors.As(err, &apiErr) {
		s := apiErr.Error()
		return strings.Contains(s, "429") || strings.Contains(strings.ToLower(s), "rate limit")
	}
	// Fallback to string matching.
	s := err.Error()
	return strings.Contains(s, "429") || strings.Contains(strings.ToLower(s), "rate limit")
}

// calculateBackoff calculates the backoff duration for retries with jitter.
func calculateBackoff(attempt int, baseDelay time.Duration) time.Duration {
	if baseDelay <= 0 {
		baseDelay = time.Second
	}
	backoff := float64(baseDelay) * math.Pow(2, float64(attempt))
	// Add jitter: random value between 0 and backoff/2.
	b := make([]byte, 1)
	if _, err := rand.Read(b); err != nil {
		return time.Duration(backoff)
	}
	jitter := float64(b[0]) / 255.0 * (backoff / 2)
	return time.Duration(backoff + jitter)
}
