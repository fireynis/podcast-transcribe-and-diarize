package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type Config struct {
	WhisperURL           string
	ChatCompletionsURL   string
	TranscriptionTimeout time.Duration
	DiarizationTimeout   time.Duration
	MaxResponseBodySize  int64
	MaxAudioFileSize     int64
}

// Segment represents a transcribed segment with timing information
type Segment struct {
	ID    int     `json:"id"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// Transcription holds the full transcription result with segments
type Transcription struct {
	Text     string    `json:"text"`
	Language string    `json:"language"`
	Duration float64   `json:"duration"`
	Segments []Segment `json:"segments"`
}

var config = Config{
	WhisperURL:           "https://api.openai.com/v1/audio/transcriptions",
	ChatCompletionsURL:   "https://api.openai.com/v1/chat/completions",
	TranscriptionTimeout: 10 * time.Minute,
	DiarizationTimeout:   5 * time.Minute,
	MaxResponseBodySize:  10 * 1024 * 1024,
	MaxAudioFileSize:     25 * 1024 * 1024,
}

// Chunking configuration
const (
	chunkDurationSec = 600 // 10 minutes per chunk
	chunkOverlapSec  = 30  // 30 seconds overlap for continuity
)

// checkFFmpeg verifies that ffmpeg is available
func checkFFmpeg() error {
	cmd := exec.Command("ffmpeg", "-version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg not found: %w (please install ffmpeg)", err)
	}
	return nil
}

// getAudioDuration returns the duration of an audio file in seconds using ffprobe
func getAudioDuration(audioPath string) (float64, error) {
	cmd := exec.Command("ffprobe",
		"-v", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1",
		audioPath)
	output, err := cmd.Output()
	if err != nil {
		return 0, fmt.Errorf("failed to get audio duration: %w", err)
	}
	var duration float64
	if _, err := fmt.Sscanf(string(output), "%f", &duration); err != nil {
		return 0, fmt.Errorf("failed to parse duration: %w", err)
	}
	return duration, nil
}

// splitAudioFile splits an audio file into chunks and returns paths to the chunk files
func splitAudioFile(audioPath string, tempDir string) ([]string, error) {
	duration, err := getAudioDuration(audioPath)
	if err != nil {
		return nil, err
	}

	var chunks []string
	startTime := 0.0
	chunkIndex := 0

	for startTime < duration {
		chunkPath := filepath.Join(tempDir, fmt.Sprintf("chunk_%03d.mp3", chunkIndex))

		// Calculate chunk duration (including overlap for non-first chunks)
		chunkStart := startTime
		if chunkIndex > 0 {
			chunkStart -= chunkOverlapSec
		}

		cmd := exec.Command("ffmpeg",
			"-i", audioPath,
			"-ss", fmt.Sprintf("%.2f", chunkStart),
			"-t", fmt.Sprintf("%d", chunkDurationSec+chunkOverlapSec),
			"-acodec", "libmp3lame",
			"-ar", "16000", // Whisper prefers 16kHz
			"-ac", "1", // Mono
			"-y", // Overwrite
			chunkPath)

		if err := cmd.Run(); err != nil {
			return nil, fmt.Errorf("failed to create chunk %d: %w", chunkIndex, err)
		}

		chunks = append(chunks, chunkPath)
		startTime += chunkDurationSec
		chunkIndex++
	}

	return chunks, nil
}

// chunkResult holds the result of transcribing a single chunk
type chunkResult struct {
	index         int
	transcription *Transcription
	startOffset   float64
	err           error
}

// transcribeChunks transcribes multiple audio chunks in parallel and stitches the results
func transcribeChunks(ctx context.Context, apiKey string, chunks []string, language string) (*Transcription, error) {
	results := make(chan chunkResult, len(chunks))
	var wg sync.WaitGroup

	// Transcribe chunks in parallel (limit concurrency to 3)
	semaphore := make(chan struct{}, 3)

	for i, chunkPath := range chunks {
		wg.Add(1)
		go func(index int, path string) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			// Calculate the start offset for this chunk
			startOffset := float64(index * chunkDurationSec)
			if index > 0 {
				startOffset -= chunkOverlapSec
			}

			trans, err := transcribeSingleFile(ctx, apiKey, path, language)
			results <- chunkResult{
				index:         index,
				transcription: trans,
				startOffset:   startOffset,
				err:           err,
			}
		}(i, chunkPath)
	}

	// Close results channel when all goroutines complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	var collected []chunkResult
	for result := range results {
		if result.err != nil {
			return nil, fmt.Errorf("chunk %d failed: %w", result.index, result.err)
		}
		collected = append(collected, result)
	}

	// Sort by index
	sort.Slice(collected, func(i, j int) bool {
		return collected[i].index < collected[j].index
	})

	// Stitch results together
	return stitchTranscriptions(collected)
}

// stitchTranscriptions combines multiple chunk transcriptions into one
func stitchTranscriptions(results []chunkResult) (*Transcription, error) {
	if len(results) == 0 {
		return nil, fmt.Errorf("no transcription results to stitch")
	}

	combined := &Transcription{
		Language: results[0].transcription.Language,
		Segments: []Segment{},
	}

	var textBuilder bytes.Buffer
	segmentID := 0
	lastEndTime := 0.0

	for i, result := range results {
		trans := result.transcription
		offset := result.startOffset

		// For chunks after the first, skip segments that overlap with previous chunk
		skipUntil := 0.0
		if i > 0 {
			skipUntil = float64(chunkOverlapSec) / 2 // Skip first half of overlap
		}

		for _, seg := range trans.Segments {
			// Skip segments in the overlap region for non-first chunks
			if i > 0 && seg.Start < skipUntil {
				continue
			}

			// Adjust timestamps
			adjustedStart := seg.Start + offset
			adjustedEnd := seg.End + offset

			// Skip if this segment would overlap with the last one
			if adjustedStart < lastEndTime-0.1 {
				continue
			}

			combined.Segments = append(combined.Segments, Segment{
				ID:    segmentID,
				Start: adjustedStart,
				End:   adjustedEnd,
				Text:  seg.Text,
			})
			segmentID++
			lastEndTime = adjustedEnd

			if textBuilder.Len() > 0 {
				textBuilder.WriteString(" ")
			}
			textBuilder.WriteString(seg.Text)
		}

		// Update duration
		if trans.Duration > 0 {
			combined.Duration = offset + trans.Duration
		}
	}

	combined.Text = textBuilder.String()
	return combined, nil
}

// transcribeSingleFile transcribes a single audio file (used for chunks)
func transcribeSingleFile(ctx context.Context, apiKey, audioPath, language string) (*Transcription, error) {
	file, err := os.Open(audioPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open audio file: %v", err)
	}
	defer file.Close()

	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	part, err := writer.CreateFormFile("file", filepath.Base(audioPath))
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %v", err)
	}
	if _, err = io.Copy(part, file); err != nil {
		return nil, fmt.Errorf("failed to copy file content: %v", err)
	}

	if err := writer.WriteField("model", "whisper-1"); err != nil {
		return nil, fmt.Errorf("failed to write model field: %v", err)
	}

	if err := writer.WriteField("response_format", "verbose_json"); err != nil {
		return nil, fmt.Errorf("failed to write response_format field: %v", err)
	}

	if language != "" {
		if err := writer.WriteField("language", language); err != nil {
			return nil, fmt.Errorf("failed to write language field: %v", err)
		}
	}

	if err = writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close writer: %v", err)
	}

	bodyBytes := requestBody.Bytes()
	req, err := http.NewRequestWithContext(ctx, "POST", config.WhisperURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(bodyBytes)), nil
	}
	req.Header.Add("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := doRequestWithRetry(ctx, req, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, config.MaxResponseBodySize))
		return nil, fmt.Errorf("non-200 response: %d, body: %s", resp.StatusCode, string(body))
	}

	var result Transcription
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	return &result, nil
}

// getOutputPaths returns transcription, diarized, and hash file paths based on flags and audio filename
func getOutputPaths(audioPath, outputDir, outputBase string) (transcriptionPath, diarizedPath, hashPath string) {
	// Determine base name from audio file if not overridden
	base := outputBase
	if base == "" {
		// Extract filename without extension from audio path
		filename := filepath.Base(audioPath)
		ext := filepath.Ext(filename)
		base = filename[:len(filename)-len(ext)]
	}

	transcriptionPath = filepath.Join(outputDir, base+"_transcription.json")
	diarizedPath = filepath.Join(outputDir, base+"_diarized.txt")
	hashPath = filepath.Join(outputDir, base+"_diarized.hash")
	return
}

// computeTranscriptionHash returns a SHA256 hash of the transcription text
func computeTranscriptionHash(text string) string {
	hash := sha256.Sum256([]byte(text))
	return hex.EncodeToString(hash[:])
}

// isDiarizationCached checks if diarization output exists and matches the current transcription
func isDiarizationCached(diarizedPath, hashPath, currentHash string) bool {
	// Check if diarized file exists
	if _, err := os.Stat(diarizedPath); os.IsNotExist(err) {
		return false
	}

	// Check if hash file exists and matches
	storedHash, err := os.ReadFile(hashPath)
	if err != nil {
		return false
	}

	return strings.TrimSpace(string(storedHash)) == currentHash
}

// httpClient with no timeout - we rely on context timeouts instead
var httpClient = &http.Client{}

// isRetryableError returns true if the error or status code indicates a retryable condition
func isRetryableError(err error, statusCode int) bool {
	if err != nil {
		// Network errors are generally retryable
		return true
	}
	// Retry on 5xx server errors and 429 rate limit
	return statusCode >= 500 || statusCode == 429
}

// doRequestWithRetry executes an HTTP request with exponential backoff retry logic
// It retries on 5xx errors, 429 rate limits, and network timeouts
// Returns the response body and any error (caller is responsible for the response)
func doRequestWithRetry(ctx context.Context, req *http.Request, maxRetries int) (*http.Response, error) {
	var lastErr error
	var resp *http.Response

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff: 1s, 2s, 4s
			backoff := time.Duration(1<<(attempt-1)) * time.Second
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
			fmt.Fprintf(os.Stderr, "Retrying request (attempt %d/%d)...\n", attempt+1, maxRetries+1)
		}

		// Clone request for retry (body needs to be re-readable)
		reqClone := req.Clone(ctx)
		if req.GetBody != nil {
			body, err := req.GetBody()
			if err != nil {
				return nil, fmt.Errorf("failed to get request body for retry: %w", err)
			}
			reqClone.Body = body
		}

		resp, lastErr = httpClient.Do(reqClone)
		if lastErr != nil {
			if isRetryableError(lastErr, 0) && attempt < maxRetries {
				continue
			}
			return nil, lastErr
		}

		if !isRetryableError(nil, resp.StatusCode) {
			return resp, nil
		}

		// Read and discard body before retry
		if resp.Body != nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}

		if attempt >= maxRetries {
			return nil, fmt.Errorf("request failed after %d retries with status %d", maxRetries+1, resp.StatusCode)
		}
	}

	return nil, lastErr
}

func main() {
	// Custom usage/help message
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `Podcast Transcription & Diarization Tool

Transcribes audio files using OpenAI Whisper and performs speaker diarization
using GPT-4 to separate different speakers in the transcript.

Usage:
  podcast-transcription -audio <file> [options]

Examples:
  podcast-transcription -audio podcast.mp3
  podcast-transcription -audio interview.mp3 -speakers 3 -names "Host,Guest1,Guest2"
  podcast-transcription -audio spanish.mp3 -language es -output-dir ./output

Options:
`)
		fmt.Fprintf(os.Stderr, "\nInput:\n")
		fmt.Fprintf(os.Stderr, "  -audio string\n\tPath to audio file (required)\n")

		fmt.Fprintf(os.Stderr, "\nOutput:\n")
		fmt.Fprintf(os.Stderr, "  -output-dir string\n\tOutput directory (default: current directory)\n")
		fmt.Fprintf(os.Stderr, "  -output string\n\tOverride base name for output files\n")

		fmt.Fprintf(os.Stderr, "\nTranscription:\n")
		fmt.Fprintf(os.Stderr, "  -language string\n\tLanguage code (e.g., 'en', 'es', 'fr'). Empty for auto-detect\n")
		fmt.Fprintf(os.Stderr, "  -no-chunk\n\tDisable automatic chunking of large files (requires ffmpeg)\n")

		fmt.Fprintf(os.Stderr, "\nDiarization:\n")
		fmt.Fprintf(os.Stderr, "  -speakers int\n\tNumber of speakers (default: 2)\n")
		fmt.Fprintf(os.Stderr, "  -names string\n\tComma-separated speaker names (e.g., 'Alice,Bob')\n")
		fmt.Fprintf(os.Stderr, "  -force-diarize\n\tForce re-diarization even if cached\n")

		fmt.Fprintf(os.Stderr, "\nEnvironment:\n")
		fmt.Fprintf(os.Stderr, "  OPENAI_API_KEY\n\tRequired. Your OpenAI API key\n")

		fmt.Fprintf(os.Stderr, "\nOutput Files:\n")
		fmt.Fprintf(os.Stderr, "  {name}_transcription.json  Full transcription with timestamps\n")
		fmt.Fprintf(os.Stderr, "  {name}_diarized.txt        Speaker-labeled transcript\n")
		fmt.Fprintf(os.Stderr, "  {name}_diarized.hash       Cache hash (auto-generated)\n")

		fmt.Fprintf(os.Stderr, "\nFor more information, see: https://github.com/your-repo/podcast-transcription\n")
	}

	// Parse command-line arguments
	audioPath := flag.String("audio", "", "Path to the audio file")
	numSpeakers := flag.Int("speakers", 2, "Number of speakers in the podcast")
	outputDir := flag.String("output-dir", ".", "Output directory for generated files")
	outputBase := flag.String("output", "", "Override base name for output files (default: derived from audio filename)")
	language := flag.String("language", "", "Language code for transcription (e.g., 'en', 'es', 'fr'). Empty for auto-detect")
	noChunk := flag.Bool("no-chunk", false, "Disable automatic chunking of large audio files")
	speakerNames := flag.String("names", "", "Comma-separated speaker names (e.g., 'Alice,Bob'). If not provided, uses 'Speaker 1', 'Speaker 2', etc.")
	forceDiarize := flag.Bool("force-diarize", false, "Force re-diarization even if cached")
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

	// Check ffmpeg availability if chunking might be needed
	if !*noChunk {
		if err := checkFFmpeg(); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: %v\nChunking will be disabled. Use -no-chunk to suppress this warning.\n", err)
			*noChunk = true
		}
	}

	// Ensure output directory exists
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output directory: %v\n", err)
		os.Exit(1)
	}

	// Calculate output paths
	transcriptionFile, diarizedFile, hashFile := getOutputPaths(*audioPath, *outputDir, *outputBase)

	var transcription *Transcription

	// Check if transcription file exists
	if _, err := os.Stat(transcriptionFile); err == nil {
		// File exists, load it
		data, err := os.ReadFile(transcriptionFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading %s: %v\n", transcriptionFile, err)
			os.Exit(1)
		}
		transcription = &Transcription{}
		if err := json.Unmarshal(data, transcription); err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing %s: %v\n", transcriptionFile, err)
			os.Exit(1)
		}
		fmt.Printf("Loaded transcription from %s\n", transcriptionFile)
	} else {
		// File doesn't exist, perform transcription
		ctx, cancel := context.WithTimeout(context.Background(), config.TranscriptionTimeout)
		defer cancel()
		var err error
		transcription, err = transcribeAudio(ctx, apiKey, *audioPath, *language, *noChunk)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error transcribing audio: %v\n", err)
			os.Exit(1)
		}

		// Save the transcription as JSON
		data, err := json.MarshalIndent(transcription, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error marshaling transcription: %v\n", err)
			os.Exit(1)
		}
		if err := os.WriteFile(transcriptionFile, data, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing transcription to file: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Transcription saved to %s\n", transcriptionFile)
	}

	// Parse speaker names if provided
	var names []string
	if *speakerNames != "" {
		names = strings.Split(*speakerNames, ",")
		for i := range names {
			names[i] = strings.TrimSpace(names[i])
		}
	}

	// Check if diarization is cached
	currentHash := computeTranscriptionHash(transcription.Text)
	if !*forceDiarize && isDiarizationCached(diarizedFile, hashFile, currentHash) {
		fmt.Printf("Diarization cached (transcription unchanged), skipping. Use -force-diarize to override.\n")
		fmt.Printf("Diarized transcript at %s\n", diarizedFile)
		return
	}

	// Diarize the transcription
	ctx, cancel := context.WithTimeout(context.Background(), config.DiarizationTimeout)
	defer cancel()
	diarizedTranscript, err := diarizeTranscript(ctx, apiKey, transcription.Text, *numSpeakers, names)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error diarizing transcript: %v\n", err)
		os.Exit(1)
	}

	// Write the diarized transcript
	if err = os.WriteFile(diarizedFile, []byte("=== Diarized Transcript ===\n"+diarizedTranscript+"\n"), 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing diarized transcript to file: %v\n", err)
		os.Exit(1)
	}

	// Save the hash for future cache checks
	if err = os.WriteFile(hashFile, []byte(currentHash), 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to write hash file: %v\n", err)
	}

	fmt.Printf("Diarized transcript saved to %s\n", diarizedFile)
}

// transcribeAudio uploads the audio file to OpenAI's Whisper API and returns the transcription with segments.
// If language is empty, Whisper will auto-detect the language.
// If noChunk is false and the file is too large, it will be split into chunks and transcribed in parallel.
func transcribeAudio(ctx context.Context, apiKey, audioPath, language string, noChunk bool) (*Transcription, error) {
	fileInfo, err := os.Stat(audioPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %v", err)
	}

	// If file is too large and chunking is enabled, split and transcribe in parallel
	if fileInfo.Size() > config.MaxAudioFileSize && !noChunk {
		fmt.Fprintf(os.Stderr, "Audio file is %d MB, splitting into chunks...\n", fileInfo.Size()/(1024*1024))

		// Create temp directory for chunks
		tempDir, err := os.MkdirTemp("", "podcast-chunks-*")
		if err != nil {
			return nil, fmt.Errorf("failed to create temp directory: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Split audio file
		chunks, err := splitAudioFile(audioPath, tempDir)
		if err != nil {
			return nil, fmt.Errorf("failed to split audio file: %v", err)
		}
		fmt.Fprintf(os.Stderr, "Created %d chunks, transcribing in parallel...\n", len(chunks))

		// Transcribe chunks in parallel
		return transcribeChunks(ctx, apiKey, chunks, language)
	}

	// File is small enough, check size limit if chunking is disabled
	if fileInfo.Size() > config.MaxAudioFileSize && noChunk {
		return nil, fmt.Errorf("audio file too large: %d bytes (max: %d bytes). Remove -no-chunk flag to enable automatic chunking", fileInfo.Size(), config.MaxAudioFileSize)
	}

	// Transcribe single file directly
	return transcribeSingleFile(ctx, apiKey, audioPath, language)
}

// diarizeTranscript sends the transcription to a ChatCompletion endpoint for speaker diarization.
// If speakerNames is provided, those names will be used instead of "Speaker 1", "Speaker 2", etc.
func diarizeTranscript(ctx context.Context, apiKey, transcript string, numSpeakers int, speakerNames []string) (string, error) {
	// Build speaker labels
	var speakerLabels string
	if len(speakerNames) > 0 {
		speakerLabels = strings.Join(speakerNames, ", ")
	} else {
		labels := make([]string, numSpeakers)
		for i := 0; i < numSpeakers; i++ {
			labels[i] = fmt.Sprintf("Speaker %d", i+1)
		}
		speakerLabels = strings.Join(labels, ", ")
	}

	var prompt string
	if len(speakerNames) > 0 {
		prompt = fmt.Sprintf(`You are an expert in speaker diarization.
Given the following transcript of a podcast with %d speakers (%s), please insert clear breaks and label each segment with the appropriate speaker name (e.g., "%s:", "%s:", etc.).

Transcript:
%s

Return the diarized transcript using the exact speaker names provided.`, numSpeakers, speakerLabels, speakerNames[0], speakerNames[min(1, len(speakerNames)-1)], transcript)
	} else {
		prompt = fmt.Sprintf(`You are an expert in speaker diarization.
Given the following transcript of a podcast and knowing there are %d speakers, please insert clear breaks and label each segment with the appropriate speaker (e.g., "Speaker 1:", "Speaker 2:", etc.).

Transcript:
%s

Return the diarized transcript.`, numSpeakers, transcript)
	}

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

	req, err := http.NewRequestWithContext(ctx, "POST", config.ChatCompletionsURL, bytes.NewReader(payloadBytes))
	if err != nil {
		return "", fmt.Errorf("failed to create chat completion request: %v", err)
	}
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(payloadBytes)), nil
	}
	req.Header.Add("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := doRequestWithRetry(ctx, req, 3)
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
