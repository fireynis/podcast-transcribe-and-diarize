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
	"regexp"
	"sort"
	"strconv"
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

// Global semaphore to limit concurrent Whisper API requests across all tracks/chunks
var whisperSemaphore = make(chan struct{}, 1)

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
			"-b:a", "32k", // Low bitrate — Whisper only needs speech quality
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

// trackFlag implements flag.Value for repeated -track flags
type trackFlag []string

func (t *trackFlag) String() string {
	return strings.Join(*t, ", ")
}

func (t *trackFlag) Set(value string) error {
	*t = append(*t, value)
	return nil
}

// TrackInput represents a parsed track: speaker name + audio file path
type TrackInput struct {
	Speaker   string
	AudioPath string
}

// SpeakerSegment is a transcribed segment tagged with a speaker name
type SpeakerSegment struct {
	Speaker string
	Segment Segment
}

// TrackTranscription pairs a speaker name with their transcription result
type TrackTranscription struct {
	Speaker       string         `json:"speaker"`
	Transcription *Transcription `json:"transcription"`
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

	// Acquire global semaphore to limit concurrent Whisper API requests
	select {
	case whisperSemaphore <- struct{}{}:
		defer func() { <-whisperSemaphore }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	req, err := http.NewRequestWithContext(ctx, "POST", config.WhisperURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(bodyBytes)), nil
	}
	req.Header.Add("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := doRequestWithRetry(ctx, req, 6)
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
	// Retry on 5xx server errors and 429 rate limit (quota errors are handled separately)
	return statusCode >= 500 || statusCode == 429
}

// isQuotaError checks if a 429 response is actually an unrecoverable quota exhaustion
func isQuotaError(body []byte) bool {
	return bytes.Contains(body, []byte("insufficient_quota"))
}

// doRequestWithRetry executes an HTTP request with exponential backoff retry logic
// It retries on 5xx errors, 429 rate limits, and network timeouts
// Returns the response body and any error (caller is responsible for the response)
func doRequestWithRetry(ctx context.Context, req *http.Request, maxRetries int) (*http.Response, error) {
	var lastErr error
	var resp *http.Response
	retryAfter := time.Duration(0)

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			var backoff time.Duration
			if retryAfter > 0 {
				// Use server-specified Retry-After with a small buffer
				backoff = retryAfter + 2*time.Second
				retryAfter = 0
			} else {
				// Exponential backoff: 2s, 4s, 8s, 16s, ...
				backoff = time.Duration(2*(1<<(attempt-1))) * time.Second
			}
			fmt.Fprintf(os.Stderr, "Retrying request (attempt %d/%d, waiting %s)...\n", attempt+1, maxRetries+1, backoff)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
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

		// Parse Retry-After header if present (429 responses)
		if resp.StatusCode == 429 {
			// Read body to check for quota exhaustion vs rate limit
			if resp.Body != nil {
				bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
				resp.Body.Close()
				if isQuotaError(bodyBytes) {
					return nil, fmt.Errorf("OpenAI quota exceeded — add credits at https://platform.openai.com/account/billing")
				}
				fmt.Fprintf(os.Stderr, "  Rate limited (429): %s\n", string(bodyBytes))
			}
			if ra := resp.Header.Get("Retry-After"); ra != "" {
				if secs, err := strconv.Atoi(ra); err == nil {
					retryAfter = time.Duration(secs) * time.Second
				}
			}
			continue
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

Transcribes audio files using OpenAI Whisper and performs speaker diarization.

Modes:
  Single-file mode:  Uses GPT-4 to identify speakers in a mixed audio file.
  Multi-track mode:  Transcribes separate per-speaker audio files and merges
                     them chronologically. Skips GPT-4 diarization entirely.

Usage:
  podcast-transcription -audio <file> [options]
  podcast-transcription -track "Speaker=file" -track "Speaker=file" [options]

Examples:
  # Single-file mode
  podcast-transcription -audio podcast.mp3
  podcast-transcription -audio interview.mp3 -speakers 3 -names "Host,Guest1,Guest2"

  # Multi-track mode
  podcast-transcription -track "Jeremy=jeremy.mp3" -track "Bob=bob.mp3"
  podcast-transcription -track "Host=host.wav" -track "Guest=guest.wav" -output episode-42

Options:
`)
		fmt.Fprintf(os.Stderr, "\nInput (mutually exclusive):\n")
		fmt.Fprintf(os.Stderr, "  -audio string\n\tPath to audio file (single-file mode)\n")
		fmt.Fprintf(os.Stderr, "  -track string\n\tSpeaker=file pair, repeatable (multi-track mode)\n")

		fmt.Fprintf(os.Stderr, "\nOutput:\n")
		fmt.Fprintf(os.Stderr, "  -output-dir string\n\tOutput directory (default: current directory)\n")
		fmt.Fprintf(os.Stderr, "  -output string\n\tOverride base name for output files\n")

		fmt.Fprintf(os.Stderr, "\nTranscription:\n")
		fmt.Fprintf(os.Stderr, "  -language string\n\tLanguage code (e.g., 'en', 'es', 'fr'). Empty for auto-detect\n")
		fmt.Fprintf(os.Stderr, "  -no-chunk\n\tDisable automatic chunking of large files (requires ffmpeg)\n")

		fmt.Fprintf(os.Stderr, "\nDiarization (single-file mode only):\n")
		fmt.Fprintf(os.Stderr, "  -speakers int\n\tNumber of speakers (default: 2)\n")
		fmt.Fprintf(os.Stderr, "  -names string\n\tComma-separated speaker names (e.g., 'Alice,Bob')\n")
		fmt.Fprintf(os.Stderr, "  -force-diarize\n\tForce re-diarization even if cached\n")

		fmt.Fprintf(os.Stderr, "\nEnvironment:\n")
		fmt.Fprintf(os.Stderr, "  OPENAI_API_KEY\n\tRequired. Your OpenAI API key\n")

		fmt.Fprintf(os.Stderr, "\nOutput Files:\n")
		fmt.Fprintf(os.Stderr, "  Single-file mode:\n")
		fmt.Fprintf(os.Stderr, "    {name}_transcription.json  Full transcription with timestamps\n")
		fmt.Fprintf(os.Stderr, "    {name}_diarized.txt        Speaker-labeled transcript\n")
		fmt.Fprintf(os.Stderr, "    {name}_diarized.hash       Cache hash (auto-generated)\n")
		fmt.Fprintf(os.Stderr, "  Multi-track mode:\n")
		fmt.Fprintf(os.Stderr, "    {speaker}_transcription.json  Per-speaker transcription (cached)\n")
		fmt.Fprintf(os.Stderr, "    {name}_merged.txt             Chronologically merged transcript\n")
		fmt.Fprintf(os.Stderr, "    {name}_merged.hash            Cache hash (auto-generated)\n")

		fmt.Fprintf(os.Stderr, "\nFor more information, see: https://github.com/your-repo/podcast-transcription\n")
	}

	// Parse command-line arguments
	var tracks trackFlag
	flag.Var(&tracks, "track", "Speaker=file pair for multi-track mode (repeatable)")
	audioPath := flag.String("audio", "", "Path to the audio file")
	numSpeakers := flag.Int("speakers", 2, "Number of speakers in the podcast")
	outputDir := flag.String("output-dir", ".", "Output directory for generated files")
	outputBase := flag.String("output", "", "Override base name for output files (default: derived from audio filename)")
	language := flag.String("language", "", "Language code for transcription (e.g., 'en', 'es', 'fr'). Empty for auto-detect")
	noChunk := flag.Bool("no-chunk", false, "Disable automatic chunking of large audio files")
	speakerNames := flag.String("names", "", "Comma-separated speaker names (e.g., 'Alice,Bob'). If not provided, uses 'Speaker 1', 'Speaker 2', etc.")
	forceDiarize := flag.Bool("force-diarize", false, "Force re-diarization even if cached")
	flag.Parse()

	hasTracks := len(tracks) > 0
	hasAudio := *audioPath != ""

	// Mutual exclusivity check
	if hasTracks && hasAudio {
		fmt.Fprintln(os.Stderr, "Error: -audio and -track are mutually exclusive. Use one or the other.")
		os.Exit(1)
	}
	if !hasTracks && !hasAudio {
		fmt.Fprintln(os.Stderr, "Error: provide either -audio <file> or -track \"Speaker=file\" (at least 2)")
		flag.Usage()
		os.Exit(1)
	}

	// Get the OpenAI API key from the environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "Please set the OPENAI_API_KEY environment variable")
		os.Exit(1)
	}

	// Multi-track mode
	if hasTracks {
		parsedTracks, err := parseTracks(tracks)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		runMultiTrackMode(apiKey, parsedTracks, *outputDir, *outputBase, *language, *noChunk)
		return
	}

	// --- Single-file mode (existing behavior, unchanged below) ---

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

	resp, err := doRequestWithRetry(ctx, req, 6)
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

// parseTracks parses and validates raw -track flag values ("Speaker=file.mp3")
func parseTracks(raw []string) ([]TrackInput, error) {
	if len(raw) < 2 {
		return nil, fmt.Errorf("at least 2 tracks are required, got %d", len(raw))
	}

	seen := make(map[string]bool)
	tracks := make([]TrackInput, 0, len(raw))

	for _, r := range raw {
		parts := strings.SplitN(r, "=", 2)
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			return nil, fmt.Errorf("invalid track format %q: expected \"Speaker=file.mp3\"", r)
		}
		speaker := parts[0]
		audioPath := parts[1]

		lowerSpeaker := strings.ToLower(speaker)
		if seen[lowerSpeaker] {
			return nil, fmt.Errorf("duplicate speaker name %q", speaker)
		}
		seen[lowerSpeaker] = true

		if _, err := os.Stat(audioPath); err != nil {
			return nil, fmt.Errorf("audio file for speaker %q not found: %v", speaker, err)
		}

		tracks = append(tracks, TrackInput{Speaker: speaker, AudioPath: audioPath})
	}

	return tracks, nil
}

// sanitizeSpeakerName returns a filesystem-safe version of a speaker name
var unsafeCharsRe = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

func sanitizeSpeakerName(name string) string {
	safe := unsafeCharsRe.ReplaceAllString(name, "_")
	return strings.ToLower(safe)
}

// getTrackTranscriptionPath returns the cache path for a per-track transcription JSON
func getTrackTranscriptionPath(outputDir, speaker string) string {
	return filepath.Join(outputDir, sanitizeSpeakerName(speaker)+"_transcription.json")
}

// getMultiTrackOutputPaths returns the merged output and hash file paths
func getMultiTrackOutputPaths(outputDir, outputBase string) (mergedPath, hashPath string) {
	base := outputBase
	if base == "" {
		base = "podcast"
	}
	mergedPath = filepath.Join(outputDir, base+"_merged.txt")
	hashPath = filepath.Join(outputDir, base+"_merged.hash")
	return
}

// computeMultiTrackHash returns a SHA256 hash of all speaker names and texts combined
func computeMultiTrackHash(trackTranscriptions []TrackTranscription) string {
	h := sha256.New()
	for _, tt := range trackTranscriptions {
		h.Write([]byte(tt.Speaker))
		h.Write([]byte(tt.Transcription.Text))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// transcribeTracksParallel transcribes all tracks concurrently (semaphore=3), reusing cached results
func transcribeTracksParallel(ctx context.Context, apiKey string, tracks []TrackInput, outputDir, language string, noChunk bool) ([]TrackTranscription, error) {
	type trackResult struct {
		index int
		tt    TrackTranscription
		err   error
	}

	results := make(chan trackResult, len(tracks))
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, 3)

	for i, track := range tracks {
		wg.Add(1)
		go func(index int, t TrackInput) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			cachePath := getTrackTranscriptionPath(outputDir, t.Speaker)

			// Try to load from cache
			if data, err := os.ReadFile(cachePath); err == nil {
				var cached Transcription
				if err := json.Unmarshal(data, &cached); err == nil {
					fmt.Printf("Loaded cached transcription for %s from %s\n", t.Speaker, cachePath)
					results <- trackResult{index: index, tt: TrackTranscription{Speaker: t.Speaker, Transcription: &cached}}
					return
				}
			}

			// Transcribe
			fmt.Printf("Transcribing track for %s (%s)...\n", t.Speaker, t.AudioPath)
			trans, err := transcribeAudio(ctx, apiKey, t.AudioPath, language, noChunk)
			if err != nil {
				results <- trackResult{index: index, err: fmt.Errorf("failed to transcribe track for %s: %w", t.Speaker, err)}
				return
			}

			// Cache the result
			data, err := json.MarshalIndent(trans, "", "  ")
			if err == nil {
				if writeErr := os.WriteFile(cachePath, data, 0644); writeErr != nil {
					fmt.Fprintf(os.Stderr, "Warning: failed to cache transcription for %s: %v\n", t.Speaker, writeErr)
				} else {
					fmt.Printf("Cached transcription for %s to %s\n", t.Speaker, cachePath)
				}
			}

			results <- trackResult{index: index, tt: TrackTranscription{Speaker: t.Speaker, Transcription: trans}}
		}(i, track)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	collected := make([]trackResult, 0, len(tracks))
	for r := range results {
		if r.err != nil {
			return nil, r.err
		}
		collected = append(collected, r)
	}

	sort.Slice(collected, func(i, j int) bool {
		return collected[i].index < collected[j].index
	})

	out := make([]TrackTranscription, len(collected))
	for i, c := range collected {
		out[i] = c.tt
	}
	return out, nil
}

// wordSet returns a set of lowercased, punctuation-stripped words from a string
func wordSet(s string) map[string]struct{} {
	words := strings.Fields(strings.ToLower(s))
	set := make(map[string]struct{}, len(words))
	for _, w := range words {
		w = strings.Trim(w, ".,;:!?\"'()-")
		if w != "" {
			set[w] = struct{}{}
		}
	}
	return set
}

// containmentSimilarity returns what fraction of the smaller set is contained in the larger one.
// This handles cases where one segment is a fragment of a longer one (common with mic bleed
// where Whisper splits the bleed into small chunks).
func containmentSimilarity(a, b map[string]struct{}) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	// Count words from the smaller set that appear in the larger set
	small, large := a, b
	if len(b) < len(a) {
		small, large = b, a
	}
	intersection := 0
	for w := range small {
		if _, ok := large[w]; ok {
			intersection++
		}
	}
	return float64(intersection) / float64(len(small))
}

// deduplicateBleeded removes segments caused by mic bleed.
// When two segments from different speakers overlap in time and share most of
// their words, the later-starting one is dropped (it's bleed from the other mic).
func deduplicateBleeded(sorted []SpeakerSegment) []SpeakerSegment {
	if len(sorted) == 0 {
		return nil
	}

	const (
		containmentThreshold = 0.6  // 60% of smaller segment's words found in larger = bleed
		timeWindowSec        = 20.0 // only compare segments starting within 20s of each other
	)

	kept := make([]bool, len(sorted))
	for i := range kept {
		kept[i] = true
	}

	for i := 0; i < len(sorted); i++ {
		if !kept[i] {
			continue
		}
		wordsI := wordSet(sorted[i].Segment.Text)

		for j := i + 1; j < len(sorted); j++ {
			if !kept[j] {
				continue
			}
			// Stop looking once segment starts are too far apart
			if sorted[j].Segment.Start-sorted[i].Segment.Start > timeWindowSec {
				break
			}
			// Only deduplicate across different speakers
			if sorted[j].Speaker == sorted[i].Speaker {
				continue
			}

			wordsJ := wordSet(sorted[j].Segment.Text)
			sim := containmentSimilarity(wordsI, wordsJ)
			if sim >= containmentThreshold {
				// Drop the later segment (it's bleed from the other mic)
				kept[j] = false
			}
		}
	}

	var result []SpeakerSegment
	dropped := 0
	for i, seg := range sorted {
		if kept[i] {
			result = append(result, seg)
		} else {
			dropped++
		}
	}
	fmt.Fprintf(os.Stderr, "Dedup: %d segments in, %d dropped, %d kept\n", len(sorted), dropped, len(result))
	return result
}

// mergeTrackSegments pools all segments from all tracks, deduplicates mic bleed,
// sorts by timestamp, and merges consecutive same-speaker segments
func mergeTrackSegments(trackTranscriptions []TrackTranscription) []SpeakerSegment {
	// Pool all segments
	var all []SpeakerSegment
	for _, tt := range trackTranscriptions {
		for _, seg := range tt.Transcription.Segments {
			all = append(all, SpeakerSegment{Speaker: tt.Speaker, Segment: seg})
		}
	}

	// Sort by Start ascending, tiebreak by End ascending
	sort.Slice(all, func(i, j int) bool {
		if all[i].Segment.Start == all[j].Segment.Start {
			return all[i].Segment.End < all[j].Segment.End
		}
		return all[i].Segment.Start < all[j].Segment.Start
	})

	// Remove mic bleed duplicates
	all = deduplicateBleeded(all)

	// Merge consecutive same-speaker segments
	if len(all) == 0 {
		return nil
	}

	merged := []SpeakerSegment{all[0]}
	for i := 1; i < len(all); i++ {
		last := &merged[len(merged)-1]
		cur := all[i]
		if cur.Speaker == last.Speaker {
			last.Segment.Text += " " + strings.TrimSpace(cur.Segment.Text)
			if cur.Segment.End > last.Segment.End {
				last.Segment.End = cur.Segment.End
			}
		} else {
			merged = append(merged, cur)
		}
	}

	return merged
}

// formatMergedTranscript formats speaker segments as "[MM:SS] Speaker: text"
func formatMergedTranscript(segments []SpeakerSegment) string {
	var b strings.Builder
	b.WriteString("=== Multi-Track Merged Transcript ===\n\n")
	for _, seg := range segments {
		mins := int(seg.Segment.Start) / 60
		secs := int(seg.Segment.Start) % 60
		fmt.Fprintf(&b, "[%02d:%02d] %s: %s\n\n", mins, secs, seg.Speaker, strings.TrimSpace(seg.Segment.Text))
	}
	return b.String()
}

// runMultiTrackMode is the top-level orchestrator for multi-track transcription and merging
func runMultiTrackMode(apiKey string, tracks []TrackInput, outputDir, outputBase, language string, noChunk bool) {
	// Check ffmpeg availability if chunking might be needed
	if !noChunk {
		if err := checkFFmpeg(); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: %v\nChunking will be disabled.\n", err)
			noChunk = true
		}
	}

	// Ensure output directory exists
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output directory: %v\n", err)
		os.Exit(1)
	}

	// Transcribe all tracks (with caching)
	ctx, cancel := context.WithTimeout(context.Background(), config.TranscriptionTimeout)
	defer cancel()

	trackTranscriptions, err := transcribeTracksParallel(ctx, apiKey, tracks, outputDir, language, noChunk)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error transcribing tracks: %v\n", err)
		os.Exit(1)
	}

	// Check merge cache
	mergedPath, hashPath := getMultiTrackOutputPaths(outputDir, outputBase)
	currentHash := computeMultiTrackHash(trackTranscriptions)
	if isDiarizationCached(mergedPath, hashPath, currentHash) {
		fmt.Printf("Merged transcript cached (transcriptions unchanged), skipping.\n")
		fmt.Printf("Merged transcript at %s\n", mergedPath)
		return
	}

	// Merge and format
	merged := mergeTrackSegments(trackTranscriptions)
	output := formatMergedTranscript(merged)

	if err := os.WriteFile(mergedPath, []byte(output), 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing merged transcript: %v\n", err)
		os.Exit(1)
	}

	if err := os.WriteFile(hashPath, []byte(currentHash), 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to write hash file: %v\n", err)
	}

	fmt.Printf("Merged transcript saved to %s\n", mergedPath)
}
