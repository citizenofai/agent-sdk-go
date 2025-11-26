package mistral

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/citizenofai/agent-sdk-go/pkg/model"
	mistralsdk "github.com/gage-technologies/mistral-go"
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

// Provider implements model.Provider for Mistral using the mistral-go SDK.
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
	client        *mistralsdk.MistralClient
	httpClient    *http.Client
	endpoint      string
	requestCount  int
	tokenCount    int
	lastResetTime time.Time
}

// NewProvider creates a new Mistral provider with sensible defaults.
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

// SetDefaultModel sets the default model for the provider.
func (p *Provider) SetDefaultModel(modelName string) *Provider {
	return p.WithDefaultModel(modelName)
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

// SetEndpoint overrides the default Mistral endpoint. Primarily intended for testing.
func (p *Provider) SetEndpoint(endpoint string) *Provider {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.endpoint = endpoint
	p.client = nil
	return p
}

// getHTTPClient returns a lazily initialised HTTP client for direct Mistral API calls.
func (p *Provider) getHTTPClient() *http.Client {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.httpClient != nil {
		return p.httpClient
	}

	// Use a sensible default timeout; rate limiting is handled separately.
	p.httpClient = &http.Client{
		Timeout: 60 * time.Second,
	}
	return p.httpClient
}

// getClient returns a lazily initialised mistral-go client.
func (p *Provider) getClient() *mistralsdk.MistralClient {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.client != nil {
		return p.client
	}

	endpoint := p.endpoint
	if endpoint == "" {
		endpoint = mistralsdk.Endpoint
	}

	p.client = mistralsdk.NewMistralClient(p.APIKey, endpoint, mistralsdk.DefaultMaxRetries, mistralsdk.DefaultTimeout)
	return p.client
}

// GetModel returns a model by name, satisfying model.Provider.
func (p *Provider) GetModel(name string) (model.Model, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if name == "" {
		name = p.DefaultModel
	}
	if name == "" {
		return nil, fmt.Errorf("mistral: no model name provided and no default model set")
	}
	if p.APIKey == "" {
		return nil, fmt.Errorf("mistral: no API key provided")
	}

	return &Model{
		ModelName: name,
		Provider:  p,
	}, nil
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
