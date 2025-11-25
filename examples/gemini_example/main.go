package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/citizenofai/agent-sdk-go/pkg/model"
	"github.com/citizenofai/agent-sdk-go/pkg/model/providers/gemini"
)

func main() {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		log.Fatal("GOOGLE_API_KEY or GEMINI_API_KEY environment variable must be set")
	}

	// Configure provider
	provider := gemini.NewProvider(apiKey).
		WithDefaultModel("gemini-2.0-flash").
		WithRateLimit(50, 100000).
		WithRetryConfig(3, 2*time.Second)

	modelInstance, err := provider.GetModel("")
	if err != nil {
		log.Fatalf("failed to get model: %v", err)
	}

	ctx := context.Background()

	// Basic non-streaming call
	req := &model.Request{
		SystemInstructions: "You are a helpful assistant.",
		Input:              "Say hello from the Gemini provider.",
	}

	resp, err := modelInstance.GetResponse(ctx, req)
	if err != nil {
		log.Fatalf("GetResponse error: %v", err)
	}

	fmt.Println("Gemini response:")
	fmt.Println(resp.Content)

	// Streaming example
	fmt.Println("\nStreaming example:")
	streamReq := &model.Request{
		Input: "Count from 1 to 5, with a short pause between each number.",
	}

	stream, err := modelInstance.StreamResponse(ctx, streamReq)
	if err != nil {
		log.Fatalf("StreamResponse error: %v", err)
	}

	for ev := range stream {
		switch ev.Type {
		case model.StreamEventTypeContent:
			fmt.Print(ev.Content)
		case model.StreamEventTypeError:
			fmt.Printf("\n[stream error] %v\n", ev.Error)
		case model.StreamEventTypeDone:
			fmt.Println("\n[stream done]")
		}
	}
}
