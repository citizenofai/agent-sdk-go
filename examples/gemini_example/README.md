# Gemini Provider Example

This example demonstrates how to use the Gemini provider with the Agent SDK Go. It shows how to configure API versioning, rate limiting, retries, and streaming.

## Prerequisites

1. A Gemini API key (see https://ai.google.dev/gemini-api/docs)
2. Go 1.23 or later

## Setup

1. Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

2. Run the example:

```bash
cd examples/gemini_example
go run main.go
```

## Features Demonstrated

- Creating and configuring a Gemini provider
- Using API versioning (default `/v1`, overridable with `SetAPIVersion`)
- Rate limiting and retry configuration
- Using tools with the agent (function calling)
- Streaming responses via `models.streamGenerateContent`

The structure and usage mirror the OpenAI example, but using the Gemini endpoints and headers (x-goog-api-key).
