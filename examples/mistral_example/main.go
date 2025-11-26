package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/citizenofai/agent-sdk-go/pkg/agent"
	"github.com/citizenofai/agent-sdk-go/pkg/model"
	"github.com/citizenofai/agent-sdk-go/pkg/model/providers/mistral"
	"github.com/citizenofai/agent-sdk-go/pkg/runner"
	"github.com/citizenofai/agent-sdk-go/pkg/tool"
)

type mistralExampleHooks struct {
	agent.DefaultAgentHooks
	toolCallCount int
}

// OnBeforeModelCall disables further tool use after a small number of successful
// tool calls so the model is forced to produce a final natural-language answer.
func (h *mistralExampleHooks) OnBeforeModelCall(ctx context.Context, a *agent.Agent, req *model.Request) error {
	// Allow up to two successful tool calls (e.g., RFC3339 and Unix) before
	// turning tools off for the remainder of the run.
	if h.toolCallCount >= 2 {
		if req.Settings == nil {
			req.Settings = &model.Settings{}
		}
		choice := "none"
		parallel := false
		req.Settings.ToolChoice = &choice
		req.Settings.ParallelToolCalls = &parallel
	}
	return nil
}

func (h *mistralExampleHooks) OnAfterToolCall(ctx context.Context, a *agent.Agent, t tool.Tool, result interface{}, err error) error {
	if err == nil {
		h.toolCallCount++
	}
	return nil
}

func (h *mistralExampleHooks) OnAgentStart(ctx context.Context, a *agent.Agent, input interface{}) error {
	h.toolCallCount = 0
	return nil
}

func main() {
	apiKey := os.Getenv("MISTRAL_API_KEY")
	if apiKey == "" {
		log.Fatal("MISTRAL_API_KEY environment variable not set")
	}

	provider := mistral.NewProvider(apiKey)
	provider.SetDefaultModel("mistral-small-latest")
	provider.WithRateLimit(50, 100000)
	provider.WithRetryConfig(3, 2*time.Second)

	fmt.Println("Mistral provider configured with:")
	fmt.Println("- Model:", "mistral-small-latest")
	fmt.Println("- Rate limit:", "50 requests/min, 100,000 tokens/min")
	fmt.Println("- Max retries:", 3)

	getCurrentTimeTool := tool.NewFunctionTool(
		"get_current_time",
		"Get the current time in a specified format",
		func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
			format := time.RFC3339

			if formatParam, ok := params["format"].(string); ok && formatParam != "" {
				switch formatParam {
				case "rfc3339":
					format = time.RFC3339
				case "kitchen":
					format = time.Kitchen
				case "date":
					format = "2006-01-02"
				case "datetime":
					format = "2006-01-02 15:04:05"
				case "unix":
					return time.Now().Unix(), nil
				}
			}

			return time.Now().Format(format), nil
		},
	).WithSchema(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"format": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"rfc3339", "kitchen", "date", "datetime", "unix"},
				"description": "The format to return the time in. Options: rfc3339, kitchen, date, datetime, unix",
			},
		},
		"required": []string{},
	})

	assistant := agent.NewAgent("Mistral Assistant")
	assistant.SetModelProvider(provider)
	assistant.WithModel("mistral-small-latest")
	// Encourage the model to use tools when they are available.
	toolChoice := "any"
	assistant.WithModelSettings(&model.Settings{ToolChoice: &toolChoice})
	assistant.SetSystemInstructions(`**Role:** You are a helpful assistant whose time-related behavior is strictly governed by the rules below.

**Behavior Rules:**

1. **Mandatory Tool Use:**
   For any user request that asks for:

   * the current time,
   * the current date,
   * the current datetime,
   * or any answer that *requires* knowing the current time,
     you **must call the "get_current_time" tool exactly once before producing your final answer.**

2. **Tool-Call Flow:**

   * Call the tool immediately after identifying a time-related question.
   * After receiving the tool result, **do not call any tools again.**
   * Use that single result to craft your final natural-language response.

3. **Timezone Assumption:**
   Assume all “current time” questions refer to the system's local timezone unless the user explicitly specifies another timezone.

4. **Formatting:**

   * Do **not** ask the user follow-up questions about format.
   * If the user gives no format, choose a reasonable, human-friendly one.
   * In your final answer, **never include raw JSON tool arguments or tool-call syntax.**
   * The final answer must be plain natural language only.

5. **Non-Time Questions:**
   If the user's request does **not** require the current time, answer normally without calling any tool.`)
	assistant.WithTools(getCurrentTimeTool)
	assistant.WithHooks(&mistralExampleHooks{})

	r := runner.NewRunner()
	r.WithDefaultProvider(provider)

	fmt.Println("\nSending a basic question to the Mistral agent...")
	result, err := r.RunSync(assistant, &runner.RunOptions{
		Input:    "Give me now the current time for the local timezone in a datetime format.",
		MaxTurns: 10,
	})
	if err != nil {
		log.Fatalf("Error running Mistral agent: %v", err)
	}

	fmt.Println("\nAgent response:")
	fmt.Println(result.FinalOutput)

	if len(result.RawResponses) > 0 {
		last := result.RawResponses[len(result.RawResponses)-1]
		if last.Usage != nil {
			fmt.Printf("\nToken usage: %d total tokens\n", last.Usage.TotalTokens)
		}
	}

	fmt.Println("\nSending a more complex question to the Mistral agent...")
	result, err = r.RunSync(assistant, &runner.RunOptions{
		Input:    "Give me now the current datetime in RFC3339 format and as a Unix timestamp.",
		MaxTurns: 10,
	})
	if err != nil {
		log.Fatalf("Error running Mistral agent: %v", err)
	}

	fmt.Println("\nAgent response:")
	fmt.Println(result.FinalOutput)

	if len(result.RawResponses) > 0 {
		last := result.RawResponses[len(result.RawResponses)-1]
		if last.Usage != nil {
			fmt.Printf("\nToken usage: %d total tokens\n", last.Usage.TotalTokens)
		}
	}

	fmt.Println("\nTesting streaming response with Mistral...")
	streamResult, err := r.RunStreaming(context.Background(), assistant, &runner.RunOptions{
		Input: "Count from 1 to 5, with a short pause between each number.",
	})
	if err != nil {
		log.Fatalf("Error running Mistral streaming: %v", err)
	}

	fmt.Println("\nStreaming response:")
	for event := range streamResult.Stream {
		switch event.Type {
		case model.StreamEventTypeContent:
			fmt.Print(event.Content)
		case model.StreamEventTypeToolCall:
			if event.ToolCall != nil {
				fmt.Printf("\n[Calling tool: %s]\n", event.ToolCall.Name)
			}
		case model.StreamEventTypeError:
			fmt.Printf("\nError: %v\n", event.Error)
		case model.StreamEventTypeDone:
			fmt.Println("\n[Done]")
		}
	}
}
