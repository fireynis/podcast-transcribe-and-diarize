# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Go-based podcast transcription tool that uses OpenAI's Whisper API for audio-to-text conversion and GPT-4 for speaker diarization. The application is designed to process podcast audio files and generate speaker-separated transcripts.

## Architecture

The application consists of a single main Go file with two primary workflows:

1. **Audio Transcription**: Uses OpenAI Whisper API to convert audio files to text
2. **Speaker Diarization**: Uses GPT-4 to identify and separate different speakers in the transcript

Key components:
- `transcribeAudio()`: Handles multipart file upload to Whisper API at main.go:84
- `diarizeTranscript()`: Processes transcript through GPT-4 for speaker separation at main.go:147
- File caching: Saves transcription.txt to avoid re-processing audio files
- Output generation: Creates diarized.txt with speaker-labeled transcript

## Development Commands

### Building and Running
```bash
# Build the application
go build -o podcast-transcription

# Run directly with Go
go run main.go -audio <path-to-audio-file> -speakers <number>

# Example usage
go run main.go -audio podcast.mp3 -speakers 2
```

### Testing and Quality
```bash
# Run tests
go test ./...

# Format code
go fmt ./...

# Lint and check for issues
go vet ./...

# Download dependencies
go mod download

# Tidy module dependencies
go mod tidy
```

## Environment Requirements

- Go 1.23 or later
- `OPENAI_API_KEY` environment variable must be set
- Audio file in supported format (mp3, wav, etc.)

## Key Dependencies

- `cloud.google.com/go/speech` - Google Cloud Speech API (unused in current implementation)
- Standard library packages for HTTP, JSON, multipart uploads
- Uses OpenAI APIs directly via HTTP requests

## File Structure

- `main.go` - Single source file containing all application logic
- `transcription.txt` - Cached transcription output (auto-generated)
- `diarized.txt` - Final diarized transcript output (auto-generated)
- `go.mod` - Module definition and dependencies
- `vendor/` - Vendored dependencies

## Command Line Interface

The application accepts two flags:
- `-audio`: Required path to audio file
- `-speakers`: Number of speakers (default: 2)

Application will skip transcription step if `transcription.txt` already exists, allowing for faster iteration on diarization.