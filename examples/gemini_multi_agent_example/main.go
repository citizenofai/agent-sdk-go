package main

import (
	"context"
	"crypto/rand"
	"fmt"
	"log"
	"math/big"
	"os"
	"strings"
	"time"

	"github.com/citizenofai/agent-sdk-go/pkg/agent"
	"github.com/citizenofai/agent-sdk-go/pkg/model/providers/gemini"
	"github.com/citizenofai/agent-sdk-go/pkg/runner"
	"github.com/citizenofai/agent-sdk-go/pkg/tool"
)

func main() {
	// Enable verbose logging
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// Get API key from environment
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		log.Fatal("GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required")
	}

	// Create a Gemini provider
	provider := gemini.NewProvider(apiKey).
		WithDefaultModel("gemini-2.0-flash").
		WithRateLimit(100, 100000).
		WithRetryConfig(3, 2*time.Second)

	fmt.Println("Provider configured with:")
	fmt.Println("- Model:", "gemini-2.0-flash")
	fmt.Println("- Rate limit:", "100 requests/min, 100,000 tokens/min")
	fmt.Println("- Max retries:", 3)

	// Create the primary agent (Frontend)
	frontendAgent := agent.NewAgent("Frontend Agent")
	frontendAgent.SetModelProvider(provider)
	frontendAgent.WithModel("gemini-2.0-flash")
	frontendAgent.SetSystemInstructions(`You are a helpful frontend agent that coordinates requests.
Your job is to understand the user's request and delegate tasks to specialized agents or use tools directly when appropriate.

IMPORTANT: For specialized tasks, you MUST delegate to the appropriate agent using the handoff mechanism:
- For mathematical calculations: DELEGATE to "Math Agent" - do NOT try to perform calculations yourself
- For weather information: DELEGATE to "Weather Agent" - do NOT try to get weather data yourself

When a user asks about:
- Any math calculation (adding, subtracting, multiplying, dividing) → handoff to "Math Agent"
- Weather conditions in any location → handoff to "Weather Agent"

You can only use the get_current_time tool directly. For all other tools, you must handoff to a specialized agent.

When you handoff to another agent, your response will be used to direct that agent. Be specific about what you're asking the specialized agent to do.

TOOL AND TURN LIMITS:
- Use at most 2 handoffs and 3 tool calls total for a single user query.
- Once you have enough information to answer the question, STOP calling tools and provide the final answer.

FINAL ANSWER REQUIREMENTS:
- Always provide a single, concise final response to the user after using tools or specialized agents.
- IMPORTANT: Never end with a tool call. Always end with a human-readable response addressed to the user.
- Do NOT output code blocks or tool invocation snippets such as "tool_code" or raw JSON showing tool calls.
- Instead, summarize what the tools or specialized agents did and clearly state the final answer in natural language.`)

	// Create the math agent
	mathAgent := agent.NewAgent("Math Agent")
	mathAgent.SetModelProvider(provider)
	mathAgent.WithModel("gemini-2.0-flash")
	mathAgent.SetSystemInstructions(`You are a specialized math agent.
You excel at solving mathematical problems and performing calculations.
Use the calculation tools available to you to solve problems accurately.

IMPORTANT WORKFLOW:
1. When you receive a request, identify the mathematical operation needed.
2. Use the appropriate calculation tool (calculate, generate_random_number, etc.) at most 2 times per query.
3. As soon as you have the result you need, STOP calling tools.
4. Provide a clear, complete answer explaining the calculation and result in one concise message.

For example, if asked to calculate 25 divided by 5, you should:
1. Use the calculate tool with operation "divide", a=25, b=5.
2. Respond once with: "The calculation of 25 divided by 5 equals 5."

RESPONSE FORMAT:
- Do NOT output code blocks or tool invocation snippets such as "tool_code" or raw JSON showing the tool call.
- Never answer by just restating the tool call (e.g., "calculate(operation=\"divide\", a=25, b=5)").
- Always answer in natural language, explaining the calculation and clearly stating the final numeric result.

Always provide educational, clear responses that explain both the process and the result.
IMPORTANT: Never end with a tool call. Always provide a final human-readable response to the user.`)

	// Create the weather agent
	weatherAgent := agent.NewAgent("Weather Agent")
	weatherAgent.SetModelProvider(provider)
	weatherAgent.WithModel("gemini-2.0-flash")
	weatherAgent.SetSystemInstructions(`You are a specialized weather agent.
You provide weather information and forecasts based on data from your tools.
Always use the available weather tools to get up-to-date information.

IMPORTANT WORKFLOW:
1. When you receive a request for weather information, quickly identify the location.
2. Use the get_weather tool at most 2 times per query.
3. Interpret the weather data and provide a complete, human-friendly response.
4. Include temperature, conditions, humidity, and any relevant context.
5. Do not keep calling tools once you can answer the question.

For example, if asked about Paris weather, you should:
1. Use the get_weather tool with location "Paris" once.
2. Interpret the data and respond with something like:
   "Currently in Paris, it's 18°C (64°F) and partly cloudy with 65% humidity. It's a pleasant day with mild temperatures."

RESPONSE FORMAT:
- Do NOT output code blocks or tool invocation snippets such as "tool_code" or raw JSON like { "get_weather": { ... } }.
- Never answer by just printing the tool call or raw data.
- Always summarize the weather in natural language (temperature, conditions, humidity, etc.) as a user-facing description.

Always provide complete, context-rich interpretations of the weather data.
IMPORTANT: Never end with a tool call. Always provide a final human-readable response to the user.`)

	// Random Number Generator Tool
	randomNumberTool := tool.NewFunctionTool(
		"generate_random_number",
		"Generate a random number between min and max (inclusive)",
		func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
			min := 1
			max := 100

			if minParam, ok := params["min"].(float64); ok {
				min = int(minParam)
			}
			if maxParam, ok := params["max"].(float64); ok {
				max = int(maxParam)
			}

			// Generate cryptographically secure random number
			delta := max - min + 1
			n, err := rand.Int(rand.Reader, big.NewInt(int64(delta)))
			if err != nil {
				return nil, fmt.Errorf("failed to generate random number: %v", err)
			}

			return min + int(n.Int64()), nil
		},
	).WithSchema(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"min": map[string]interface{}{
				"type":        "integer",
				"description": "The minimum value (inclusive)",
			},
			"max": map[string]interface{}{
				"type":        "integer",
				"description": "The maximum value (inclusive)",
			},
		},
		"required": []string{},
	})

	// Simple Calculator Tool
	calculatorTool := tool.NewFunctionTool(
		"calculate",
		"Perform a simple calculation",
		func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
			operation, ok := params["operation"].(string)
			if !ok {
				return nil, fmt.Errorf("operation parameter is required")
			}

			a, aOk := params["a"].(float64)
			b, bOk := params["b"].(float64)

			if !aOk || !bOk {
				return nil, fmt.Errorf("both 'a' and 'b' parameters are required and must be numbers")
			}

			switch strings.ToLower(operation) {
			case "add":
				return a + b, nil
			case "subtract":
				return a - b, nil
			case "multiply":
				return a * b, nil
			case "divide":
				if b == 0 {
					return nil, fmt.Errorf("division by zero is not allowed")
				}
				return a / b, nil
			default:
				return nil, fmt.Errorf("unknown operation: %s", operation)
			}
		},
	).WithSchema(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"add", "subtract", "multiply", "divide"},
				"description": "The mathematical operation to perform",
			},
			"a": map[string]interface{}{
				"type":        "number",
				"description": "The first operand",
			},
			"b": map[string]interface{}{
				"type":        "number",
				"description": "The second operand",
			},
		},
		"required": []string{"operation", "a", "b"},
	})

	// Weather Tool (mock)
	weatherTool := tool.NewFunctionTool(
		"get_weather",
		"Get the current weather for a location (mocked data)",
		func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
			location, ok := params["location"].(string)
			if !ok || location == "" {
				return nil, fmt.Errorf("location parameter is required")
			}

			// Mock weather data
			conditions := []string{"sunny", "partly cloudy", "cloudy", "rainy", "stormy", "snowy", "foggy", "windy"}
			temps := []int{-10, 0, 5, 10, 15, 20, 25, 30, 35}

			condIdx, err := rand.Int(rand.Reader, big.NewInt(int64(len(conditions))))
			if err != nil {
				return nil, fmt.Errorf("failed to generate random weather condition: %v", err)
			}
			tempIdx, err := rand.Int(rand.Reader, big.NewInt(int64(len(temps))))
			if err != nil {
				return nil, fmt.Errorf("failed to generate random temperature: %v", err)
			}
			humidityBig, err := rand.Int(rand.Reader, big.NewInt(100))
			if err != nil {
				return nil, fmt.Errorf("failed to generate random humidity: %v", err)
			}

			return map[string]interface{}{
				"location":    location,
				"condition":   conditions[condIdx.Int64()],
				"temperature": temps[tempIdx.Int64()],
				"humidity":    int(humidityBig.Int64()),
				"unit":        "celsius",
				"timestamp":   time.Now().Format(time.RFC3339),
			}, nil
		},
	).WithSchema(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"location": map[string]interface{}{
				"type":        "string",
				"description": "The location to get weather for (city name)",
			},
		},
		"required": []string{"location"},
	})

	// Time Tool (shared by all agents)
	timeTool := tool.NewFunctionTool(
		"get_current_time",
		"Get the current time in a specified format. This tool will return the current system time, not the time in a specific location.",
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

	// Attach tools to agents
	frontendAgent.WithTools(timeTool)

	mathAgent.WithTools(calculatorTool)
	mathAgent.WithTools(randomNumberTool)
	mathAgent.WithTools(timeTool)

	weatherAgent.WithTools(weatherTool)
	weatherAgent.WithTools(timeTool)

	// Set up handoffs: frontend can delegate to math and weather agents
	frontendAgent.WithHandoffs(mathAgent, weatherAgent)

	// Create a runner
	r := runner.NewRunner()
	r.WithDefaultProvider(provider)

	// Run example with a math query
	fmt.Println("Running with a math query...")
	result, err := r.RunSync(frontendAgent, &runner.RunOptions{
		Input:    "What is 42 divided by 6?",
		MaxTurns: 10,
	})
	if err != nil {
		log.Fatalf("Error running agent: %v", err)
	}

	fmt.Println("\nAgent response:")
	fmt.Println(result.FinalOutput)
	fmt.Println("\nItems generated:", len(result.NewItems))
	fmt.Printf("Run stats: model_calls=%d, MaxTurns=%d, FinalOutputNil=%v\n", len(result.RawResponses), 10, result.FinalOutput == nil)

	fmt.Println("\nDetailed items:")
	for i, item := range result.NewItems {
		fmt.Printf("Item %d: Type=%s\n", i, item.GetType())
	}

	// Run example with a weather query
	fmt.Println("\nRunning with a weather query...")
	result, err = r.RunSync(frontendAgent, &runner.RunOptions{
		Input:    "What's the current weather in Paris?",
		MaxTurns: 10,
	})
	if err != nil {
		log.Fatalf("Error running agent: %v", err)
	}

	fmt.Println("\nAgent response:")
	fmt.Println(result.FinalOutput)
	fmt.Println("\nItems generated:", len(result.NewItems))
	fmt.Printf("Run stats: model_calls=%d, MaxTurns=%d, FinalOutputNil=%v\n", len(result.RawResponses), 10, result.FinalOutput == nil)

	fmt.Println("\nDetailed items:")
	for i, item := range result.NewItems {
		fmt.Printf("Item %d: Type=%s\n", i, item.GetType())
	}

	// Run example with a mixed query
	fmt.Println("\nRunning with a mixed query...")
	result, err = r.RunSync(frontendAgent, &runner.RunOptions{
		Input:    "What is 15 x 4 and what's the current time?",
		MaxTurns: 10,
	})
	if err != nil {
		log.Fatalf("Error running agent: %v", err)
	}

	fmt.Println("\nAgent response:")
	fmt.Println(result.FinalOutput)
	fmt.Println("\nItems generated:", len(result.NewItems))
	fmt.Printf("Run stats: model_calls=%d, MaxTurns=%d, FinalOutputNil=%v\n", len(result.RawResponses), 10, result.FinalOutput == nil)

	fmt.Println("\nDetailed items:")
	for i, item := range result.NewItems {
		fmt.Printf("Item %d: Type=%s\n", i, item.GetType())
	}
}
