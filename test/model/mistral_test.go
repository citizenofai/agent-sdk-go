package model_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/citizenofai/agent-sdk-go/pkg/model"
	"github.com/citizenofai/agent-sdk-go/pkg/model/providers/mistral"
	"github.com/stretchr/testify/assert"
)

func TestMistralProvider_NewProvider(t *testing.T) {
	p := mistral.NewProvider("test-key")
	assert.Equal(t, "test-key", p.APIKey)
}

func TestMistralProvider_GetModel(t *testing.T) {
	p := mistral.NewProvider("test-key").WithDefaultModel("mistral-small-latest")

	m, err := p.GetModel("mistral-large-latest")
	assert.NoError(t, err)
	assert.Equal(t, "mistral-large-latest", m.(*mistral.Model).ModelName)

	m, err = p.GetModel("")
	assert.NoError(t, err)
	assert.Equal(t, "mistral-small-latest", m.(*mistral.Model).ModelName)
}

func TestMistralProvider_GetModel_NoAPIKey(t *testing.T) {
	p := mistral.NewProvider("")
	_, err := p.GetModel("mistral-small-latest")
	assert.Error(t, err)
}

func TestMistralModel_GetResponse_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var body map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&body)
		assert.NoError(t, err)

		resp := map[string]interface{}{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   "mistral-small-latest",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Test response",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]interface{}{
				"prompt_tokens":     5,
				"completion_tokens": 7,
				"total_tokens":      12,
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := mistral.NewProvider("test-key").WithDefaultModel("mistral-small-latest")
	p.SetEndpoint(server.URL)

	m, err := p.GetModel("")
	assert.NoError(t, err)

	req := &model.Request{
		SystemInstructions: "You are a test assistant.",
		Input:              "Say hello",
	}

	resp, err := m.GetResponse(context.Background(), req)
	assert.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "Test response", resp.Content)
	assert.Equal(t, 12, resp.Usage.TotalTokens)
}

func TestMistralModel_GetResponse_WithTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&body)

		tools, ok := body["tools"].([]interface{})
		assert.True(t, ok)
		assert.Len(t, tools, 1)

		resp := map[string]interface{}{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   "mistral-small-latest",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role": "assistant",
						"tool_calls": []map[string]interface{}{
							{
								"id":   "tool_1",
								"type": "function",
								"function": map[string]interface{}{
									"name":      "test_tool",
									"arguments": `{"param1":"value1"}`,
								},
							},
						},
					},
					"finish_reason": "tool_calls",
				},
			},
			"usage": map[string]interface{}{
				"total_tokens": 10,
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := mistral.NewProvider("test-key").WithDefaultModel("mistral-small-latest")
	p.SetEndpoint(server.URL)
	m, err := p.GetModel("")
	assert.NoError(t, err)

	toolDef := map[string]interface{}{
		"name":        "test_tool",
		"description": "A test tool",
		"parameters": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"param1": map[string]interface{}{"type": "string"},
			},
		},
	}

	req := &model.Request{
		Input: "Test input",
		Tools: []interface{}{toolDef},
	}

	resp, err := m.GetResponse(context.Background(), req)
	assert.NoError(t, err)
	assert.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, "test_tool", resp.ToolCalls[0].Name)
	assert.Equal(t, "value1", resp.ToolCalls[0].Parameters["param1"])
}
