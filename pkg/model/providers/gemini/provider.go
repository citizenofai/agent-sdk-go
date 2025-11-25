package gemini

import (
	"context"
	"fmt"
	"sync"
	"time"

	genai "google.golang.org/genai"

	"github.com/citizenofai/agent-sdk-go/pkg/model"
)

const (
	// DefaultRPM is the default rate limit for requests per minute.
	DefaultRPM = 200

	// DefaultTPM is the default rate limit for tokens per minute.
	DefaultTPM = 150000

	// DefaultMaxRetries is the default number of retries for transient / rate-limit errors.
	DefaultMaxRetries = 5

	// DefaultRetryAfter is the default base delay before retrying a failed request.
	DefaultRetryAfter = 1 * time.Second
)

// Provider implements model.Provider for Gemini using the official go-genai SDK.
type Provider struct {
	// Configuration
	APIKey       string
	DefaultModel string

	// Rate limiting configuration
	RPM        int           // Requests per minute
	TPM        int           // Tokens per minute
	MaxRetries int           // Maximum number of retries
	RetryAfter time.Duration // Base delay before retrying

	// Internal state
	mu            sync.Mutex
	client        *genai.Client
	requestCount  int
	tokenCount    int
	lastResetTime time.Time
}

// NewProvider creates a new Gemini provider with sensible defaults.
func NewProvider(apiKey string) *Provider {
	return &Provider{
		APIKey:        apiKey,
		RPM:           DefaultRPM,
		TPM:           DefaultTPM,
		MaxRetries:    DefaultMaxRetries,
		RetryAfter:    DefaultRetryAfter,
		lastResetTime: time.Now(),
	}
}

// WithDefaultModel sets the default model for the provider.
func (p *Provider) WithDefaultModel(modelName string) *Provider {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.DefaultModel = modelName
	return p
}

// WithRateLimit configures request/token rate limits for the provider.
func (p *Provider) WithRateLimit(rpm, tpm int) *Provider {
	p.mu.Lock()
	defer p.mu.Unlock()
	if rpm > 0 {
		p.RPM = rpm
	}
	if tpm > 0 {
		p.TPM = tpm
	}
	return p
}

// WithRetryConfig configures retry behaviour for the provider.
func (p *Provider) WithRetryConfig(maxRetries int, retryAfter time.Duration) *Provider {
	p.mu.Lock()
	defer p.mu.Unlock()
	if maxRetries >= 0 {
		p.MaxRetries = maxRetries
	}
	if retryAfter > 0 {
		p.RetryAfter = retryAfter
	}
	return p
}

// SetDefaultModel sets the default model for the provider
func (p *Provider) SetDefaultModel(modelName string) *Provider {
	return p.WithDefaultModel(modelName)
}

// GetModel returns a model by name, satisfying model.Provider.
func (p *Provider) GetModel(name string) (model.Model, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if name == "" {
		name = p.DefaultModel
	}
	if name == "" {
		return nil, fmt.Errorf("gemini: no model name provided and no default model set")
	}
	if p.APIKey == "" {
		return nil, fmt.Errorf("gemini: no API key provided")
	}

	return &Model{
		ModelName: name,
		Provider:  p,
	}, nil
}

// getClient returns a lazily initialised genai.Client.
func (p *Provider) getClient(ctx context.Context) (*genai.Client, error) {
	if p.client != nil {
		return p.client, nil
	}

	cfg := &genai.ClientConfig{
		APIKey:  p.APIKey,
		Backend: genai.BackendGeminiAPI,
	}
	client, err := genai.NewClient(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("gemini: failed to create client: %w", err)
	}
	p.client = client
	return client, nil
}

// WaitForRateLimit blocks until another request is allowed under the configured RPM/TPM.
func (p *Provider) WaitForRateLimit() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if time.Since(p.lastResetTime) >= time.Minute {
		p.requestCount = 0
		p.tokenCount = 0
		p.lastResetTime = time.Now()
	}

	if p.RPM <= 0 && p.TPM <= 0 {
		return
	}

	if (p.RPM > 0 && p.requestCount >= p.RPM) || (p.TPM > 0 && p.tokenCount >= p.TPM) {
		var wait time.Duration
		if p.RPM > 0 && p.requestCount >= p.RPM {
			wait = time.Minute / time.Duration(p.RPM)
		}
		if p.TPM > 0 && p.tokenCount >= p.TPM {
			t := time.Minute / time.Duration(p.TPM)
			if t > wait {
				wait = t
			}
		}
		time.Sleep(wait)
	}

	p.requestCount++
}

// UpdateTokenCount increments the tracked token usage for rate limiting.
func (p *Provider) UpdateTokenCount(tokens int) {
	if tokens <= 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.tokenCount += tokens
}
