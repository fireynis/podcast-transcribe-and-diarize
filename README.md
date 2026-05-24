# Podcast Transcription & Diarization Tool

A Go-based tool that transcribes podcast audio files using OpenAI's Whisper API and performs speaker diarization using GPT-4 to separate different speakers in the transcript.

## Features

- **Audio Transcription**: Uses OpenAI Whisper API for accurate speech-to-text conversion
- **Speaker Diarization**: Leverages GPT-4 to identify and label different speakers
- **Automatic Chunking**: Splits large audio files (>25MB) into chunks and processes in parallel
- **Timestamp Support**: Captures word-level timestamps for subtitle generation
- **Named Speakers**: Optionally provide speaker names instead of generic labels
- **Smart Caching**: Caches both transcription and diarization results
- **Multiple Languages**: Supports language specification or auto-detection
- **Retry Logic**: Automatic retry with exponential backoff for API failures
- **Flexible Output**: Configurable output directory and file naming

## Prerequisites

- Go 1.23 or later
- OpenAI API key with access to Whisper and GPT-4
- ffmpeg (optional, required for chunking large audio files)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd podcast-transcription
```

2. Build the application:
```bash
go build -o podcast-transcription
```

3. (Optional) Install ffmpeg for large file support:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (with chocolatey)
choco install ffmpeg
```

## Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Quick Start

```bash
# Show help
./podcast-transcription -help

# Basic usage with 2 speakers (default)
./podcast-transcription -audio podcast.mp3

# Specify number of speakers
./podcast-transcription -audio interview.wav -speakers 3

# Use named speakers
./podcast-transcription -audio podcast.mp3 -names "Alice,Bob"
```

### Command Line Options

```
podcast-transcription [options]

Input:
  -audio string       Path to audio file (required)

Output:
  -output-dir string  Output directory (default: current directory)
  -output string      Override base name for output files

Transcription:
  -language string    Language code (e.g., 'en', 'es', 'fr'). Empty for auto-detect
  -no-chunk           Disable automatic chunking of large files

Diarization:
  -speakers int       Number of speakers (default: 2)
  -names string       Comma-separated speaker names (e.g., 'Alice,Bob')
  -force-diarize      Force re-diarization even if cached
```

### Examples

```bash
# Basic transcription with 2 speakers
./podcast-transcription -audio podcast.mp3

# Interview with named speakers
./podcast-transcription -audio interview.mp3 -speakers 2 -names "Host,Guest"

# Spanish podcast with custom output directory
./podcast-transcription -audio spanish-podcast.mp3 -language es -output-dir ./output

# Force re-diarization of cached transcription
./podcast-transcription -audio podcast.mp3 -force-diarize

# Large file without chunking (will fail if >25MB)
./podcast-transcription -audio large-podcast.mp3 -no-chunk

# Custom output naming
./podcast-transcription -audio recording.mp3 -output "episode-42"
# Creates: episode-42_transcription.json, episode-42_diarized.txt
```

## Output Files

The tool generates the following output files (using audio filename as base):

| File | Description |
|------|-------------|
| `{name}_transcription.json` | Full transcription with timestamps and segments |
| `{name}_diarized.txt` | Speaker-labeled transcript |
| `{name}_diarized.hash` | Cache hash for diarization (auto-generated) |

### Transcription JSON Format

```json
{
  "text": "Full transcript text...",
  "language": "en",
  "duration": 1234.56,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Segment text..."
    }
  ]
}
```

## Caching Behavior

- **Transcription**: Cached in `*_transcription.json`. Delete to force re-transcription.
- **Diarization**: Cached based on transcription hash. Automatically re-runs if transcription changes. Use `-force-diarize` to override.

## Large File Handling

Files over 25MB are automatically split into 10-minute chunks with 30-second overlap:

1. Audio is split using ffmpeg
2. Chunks are transcribed in parallel (3 concurrent)
3. Results are stitched with timestamp adjustment
4. Overlap regions are deduplicated

Use `-no-chunk` to disable this behavior (will error if file exceeds 25MB).

## Configuration Defaults

| Setting | Value |
|---------|-------|
| Transcription Timeout | 10 minutes |
| Diarization Timeout | 5 minutes |
| Max Audio File Size | 25MB (per chunk) |
| Max Response Body Size | 10MB |
| Chunk Duration | 10 minutes |
| Chunk Overlap | 30 seconds |

## Supported Audio Formats

Any format supported by OpenAI Whisper:
- MP3, WAV, M4A, FLAC, OGG, WebM

## Error Handling

The tool includes robust error handling:
- **Retry Logic**: Automatic retry with exponential backoff (1s, 2s, 4s) for 5xx errors and rate limits
- **Graceful Degradation**: Falls back to non-chunking mode if ffmpeg unavailable
- **Validation**: File size and format validation before upload

## Troubleshooting

### Common Issues

**"Please set the OPENAI_API_KEY environment variable"**
- Ensure your OpenAI API key is properly set

**"audio file too large"**
- Install ffmpeg to enable automatic chunking, or use `-no-chunk` with smaller files

**"ffmpeg not found"**
- Install ffmpeg for large file support, or use `-no-chunk` flag

**Timeout errors**
- Check internet connection
- Large files may need longer processing time
- The tool will automatically retry on transient failures

**Rate limit errors (429)**
- The tool automatically retries with backoff
- Consider reducing parallel chunk processing

## Development

### Building from Source

```bash
go mod download
go build -o podcast-transcription
```

### Running Tests

```bash
go test ./...
```

### Code Quality

```bash
go fmt ./...
go vet ./...
```

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]
