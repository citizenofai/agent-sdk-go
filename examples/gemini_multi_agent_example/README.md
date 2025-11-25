# Gemini Multi-Agent Example

This example demonstrates how to use the **Gemini provider** with the Agent SDK Go to build a **multi-agent system with handoffs**.

It mirrors the functionality of the existing multi-agent examples (LM Studio and Anthropic) but uses the official `google.golang.org/genai`-backed Gemini provider.

## Prerequisites

1. A Gemini API key for the Gemini Developer API.
2. Go 1.23 or later.

## Setup

1. Set your Gemini API key as an environment variable (either works):

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
# or
export GEMINI_API_KEY="your-gemini-api-key"
```

2. Run the example:

```bash
cd examples/gemini_multi_agent_example
go run .
```

## What This Example Shows

- A **frontend agent** that routes user requests.
- A **Math Agent** for calculations using tools.
- A **Weather Agent** that returns mocked weather data via tools.
- **Handoffs** between agents using the same conventions as the OpenAI/Anthropic providers (via `WithHandoffs`).
- Tools defined with `tool.NewFunctionTool` and JSON schemas, reused by multiple agents.

The frontend agent decides when to:

- Delegate math questions to the Math Agent.
- Delegate weather questions to the Weather Agent.
- Use a shared `get_current_time` tool directly.

The runner and Gemini provider work together so that:

- Handoff tools (e.g. `handoff_to_[Agent]`) are exposed to Gemini as function tools.
- Gemini's function calling triggers `ToolCall` and `HandoffCall` events.
- The runner routes control between agents based on these handoffs.

This gives you a concrete template for building **multi-agent workflows** on top of the Gemini provider.
