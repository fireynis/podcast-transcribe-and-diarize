package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

type Config struct {
	WhisperURL           string
	ChatCompletionsURL   string
	TranscriptionFile    string
	DiarizedFile         string
	TranscriptionTimeout time.Duration
	DiarizationTimeout   time.Duration
	MaxResponseBodySize  int64
	MaxAudioFileSize     int64
	HTTPTimeout          time.Duration
}

var config = Config{
	WhisperURL:           "https://api.openai.com/v1/audio/transcriptions",
	ChatCompletionsURL:   "https://api.openai.com/v1/chat/completions",
	TranscriptionFile:    "transcription.txt",
	DiarizedFile:         "diarized.txt",
	TranscriptionTimeout: 5 * time.Minute,
	DiarizationTimeout:   2 * time.Minute,
	MaxResponseBodySize:  10 * 1024 * 1024,
	MaxAudioFileSize:     25 * 1024 * 1024,
	HTTPTimeout:          30 * time.Second,
}

var httpClient = &http.Client{
	Timeout: config.HTTPTimeout,
}

func main() {
	// Parse command-line arguments
	audioPath := flag.String("audio", "", "Path to the audio file")
	numSpeakers := flag.Int("speakers", 2, "Number of speakers in the podcast")
	flag.Parse()

	if *audioPath == "" {
		fmt.Fprintln(os.Stderr, "Please provide the path to the audio file using -audio")
		os.Exit(1)
	}

	// Get the OpenAI API key from the environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "Please set the OPENAI_API_KEY environment variable")
		os.Exit(1)
	}

	var transcript string

	// Check if transcription.txt exists
	if _, err := os.Stat(config.TranscriptionFile); err == nil {
		// File exists, load it
		data, err := os.ReadFile(config.TranscriptionFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading %s: %v\n", config.TranscriptionFile, err)
			os.Exit(1)
		}
		transcript = string(data)
		fmt.Printf("Loaded transcription from %s\n", config.TranscriptionFile)
	} else {
		// File doesn't exist, perform transcription
		ctx, cancel := context.WithTimeout(context.Background(), config.TranscriptionTimeout)
		defer cancel()
		transcript, err = transcribeAudio(ctx, apiKey, *audioPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error transcribing audio: %v\n", err)
			os.Exit(1)
		}

		// Save the transcription to transcription.txt
		if err := os.WriteFile(config.TranscriptionFile, []byte(transcript), 0644); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing transcription to file: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Transcription saved to %s\n", config.TranscriptionFile)
	}

	// Diarize the transcription using the o1 model
	ctx, cancel := context.WithTimeout(context.Background(), config.DiarizationTimeout)
	defer cancel()
	diarizedTranscript, err := diarizeTranscript(ctx, apiKey, transcript, *numSpeakers)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error diarizing transcript: %v\n", err)
		os.Exit(1)
	}

	// Write the diarized transcript to diarized.txt
	if err = os.WriteFile(config.DiarizedFile, []byte("=== Diarized Transcript ===\n"+diarizedTranscript+"\n"), 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing diarized transcript to file: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Diarized transcript saved to %s\n", config.DiarizedFile)
}

// transcribeAudio uploads the audio file to OpenAI's Whisper API and returns the transcription text.
func transcribeAudio(ctx context.Context, apiKey, audioPath string) (string, error) {
	fileInfo, err := os.Stat(audioPath)
	if err != nil {
		return "", fmt.Errorf("failed to get file info: %v", err)
	}
	if fileInfo.Size() > config.MaxAudioFileSize {
		return "", fmt.Errorf("audio file too large: %d bytes (max: %d bytes)", fileInfo.Size(), config.MaxAudioFileSize)
	}

	file, err := os.Open(audioPath)
	if err != nil {
		return "", fmt.Errorf("failed to open audio file: %v", err)
	}
	defer func() {
		if cerr := file.Close(); cerr != nil {
			fmt.Fprintf(os.Stderr, "Error closing audio file: %v\n", cerr)
		}
	}()

	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	part, err := writer.CreateFormFile("file", filepath.Base(audioPath))
	if err != nil {
		return "", fmt.Errorf("failed to create form file: %v", err)
	}
	if _, err = io.Copy(part, file); err != nil {
		return "", fmt.Errorf("failed to copy file content: %v", err)
	}

	if err := writer.WriteField("model", "whisper-1"); err != nil {
		return "", fmt.Errorf("failed to write model field: %v", err)
	}

	if err = writer.Close(); err != nil {
		return "", fmt.Errorf("failed to close writer: %v", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", config.WhisperURL, &requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Add("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %v", err)
	}
	defer func() {
		if cerr := resp.Body.Close(); cerr != nil {
			fmt.Fprintf(os.Stderr, "Error closing transcription response body: %v\n", cerr)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, config.MaxResponseBodySize))
		return "", fmt.Errorf("non-200 response: %d, body: %s", resp.StatusCode, string(body))
	}

	var res struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return "", fmt.Errorf("failed to decode response: %v", err)
	}
	return res.Text, nil
}

// diarizeTranscript sends the transcription to a ChatCompletion endpoint using the o1 model.
// It does not set a maximum token limit in the request.
func diarizeTranscript(ctx context.Context, apiKey, transcript string, numSpeakers int) (string, error) {
	prompt := fmt.Sprintf(`You are an expert in speaker diarization.
Given the following transcript of a podcast and knowing there are %d speakers, please insert clear breaks and label each segment with the appropriate speaker (e.g., "Speaker 1:", "Speaker 2:", etc.).

Transcript:
%s

Return the diarized transcript.`, numSpeakers, transcript)

	payload := map[string]interface{}{
		"model":       "gpt-4o",
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
		"temperature": 0.3,
		// "max_tokens" is intentionally omitted to allow the API to use the model's full output capacity.
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal payload: %v", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", config.ChatCompletionsURL, bytes.NewBuffer(payloadBytes))
	if err != nil {
		return "", fmt.Errorf("failed to create chat completion request: %v", err)
	}
	req.Header.Add("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send chat completion request: %v", err)
	}
	defer func() {
		if cerr := resp.Body.Close(); cerr != nil {
			fmt.Fprintf(os.Stderr, "Error closing chat completion response body: %v\n", cerr)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, config.MaxResponseBodySize))
		return "", fmt.Errorf("non-200 response from chat completion: %d, body: %s", resp.StatusCode, string(body))
	}

	var res struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return "", fmt.Errorf("failed to decode chat completion response: %v", err)
	}

	if len(res.Choices) == 0 {
		return "", fmt.Errorf("no choices returned from chat completion")
	}
	return res.Choices[0].Message.Content, nil
}
