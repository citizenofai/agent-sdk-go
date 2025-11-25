package model_test

import (
	"testing"
	"time"

	"github.com/citizenofai/agent-sdk-go/pkg/model"
	"github.com/citizenofai/agent-sdk-go/pkg/model/providers/gemini"
	"github.com/stretchr/testify/assert"
)

func TestGeminiProvider_NewProvider(t *testing.T) {
	p := gemini.NewProvider("test-key")
	assert.Equal(t, "test-key", p.APIKey)
	assert.Equal(t, gemini.DefaultMaxRetries, p.MaxRetries)
	assert.Equal(t, gemini.DefaultRPM, p.RPM)
	assert.Equal(t, gemini.DefaultTPM, p.TPM)
}

func TestGeminiProvider_GetModel(t *testing.T) {
	p := gemini.NewProvider("test-key").WithDefaultModel("gemini-2.0-flash")

	// Explicit model name
	m, err := p.GetModel("gemini-2.0-pro")
	assert.NoError(t, err)
	assert.Equal(t, "gemini-2.0-pro", m.(*gemini.Model).ModelName)

	// Default model
	m, err = p.GetModel("")
	assert.NoError(t, err)
	assert.Equal(t, "gemini-2.0-flash", m.(*gemini.Model).ModelName)
}

func TestGeminiProvider_GetModel_NoAPIKey(t *testing.T) {
	p := gemini.NewProvider("")
	_, err := p.GetModel("gemini-2.0-flash")
	assert.Error(t, err)
}

func TestGeminiProvider_RateLimiting(t *testing.T) {
	p := gemini.NewProvider("test-key")
	p.WithRateLimit(2, 1000) // 2 requests per minute

	// First two requests should not incur noticeable delay
	start := time.Now()
	p.WaitForRateLimit()
	p.WaitForRateLimit()
	assert.Less(t, time.Since(start), 100*time.Millisecond)

	// Third request should sleep at least a little
	start = time.Now()
	p.WaitForRateLimit()
	assert.Greater(t, time.Since(start), 0*time.Millisecond)
}

func TestGeminiProvider_UpdateTokenCount(t *testing.T) {
	p := gemini.NewProvider("test-key")
	p.WithRateLimit(100, 10) // very low TPM

	// Bump token usage past TPM, then WaitForRateLimit should sleep
	p.UpdateTokenCount(20)
	start := time.Now()
	p.WaitForRateLimit()
	assert.Greater(t, time.Since(start), 0*time.Millisecond)
}

func TestGeminiBuildInputText(t *testing.T) {
	t.Run("NilRequest", func(t *testing.T) {
		text := gemini.TestBuildInputText(nil)
		assert.Equal(t, "", text)
	})

	t.Run("StringInput", func(t *testing.T) {
		text := gemini.TestBuildInputText(&model.Request{Input: "hello"})
		assert.Equal(t, "hello", text)
	})

	t.Run("SystemAndMessages", func(t *testing.T) {
		req := &model.Request{
			SystemInstructions: "You are a test assistant.",
			Input: []interface{}{
				map[string]interface{}{"content": "First"},
				map[string]interface{}{"content": "Second"},
			},
		}
		text := gemini.TestBuildInputText(req)
		assert.Equal(t, "You are a test assistant.\n\n\nFirst\nSecond", text)
	})
}
