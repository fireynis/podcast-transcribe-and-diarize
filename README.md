# Podcast Transcription & Diarization Tool

A Go-based tool that transcribes podcast audio files using OpenAI's Whisper API and performs speaker diarization using GPT-4 to separate different speakers in the transcript.

## Features

- **Audio Transcription**: Uses OpenAI Whisper API for accurate speech-to-text conversion
- **Speaker Diarization**: Leverages GPT-4 to identify and label different speakers
- **Smart Caching**: Saves transcription results to avoid re-processing audio files
- **Timeout Protection**: Configurable timeouts prevent hanging on network issues
- **File Size Validation**: Validates audio file size before upload (25MB limit)
- **Memory Protection**: Limits response body reads to prevent memory exhaustion

## Prerequisites

- Go 1.23 or later
- OpenAI API key with access to Whisper and GPT-4

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

## Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```bash
./podcast-transcription -audio <path-to-audio-file> -speakers <number-of-speakers>
```

### Examples

```bash
# Transcribe a podcast with 2 speakers (default)
./podcast-transcription -audio podcast.mp3

# Transcribe a podcast with 3 speakers
./podcast-transcription -audio interview.wav -speakers 3

# Run with Go directly
go run main.go -audio podcast.mp3 -speakers 2
```

### Command Line Options

- `-audio` (required): Path to the audio file (supports mp3, wav, and other formats supported by Whisper)
- `-speakers` (optional): Number of speakers in the podcast (default: 2)

## Output Files

The tool generates two output files:

1. **`transcription.txt`**: Raw transcription from Whisper API
   - Cached to avoid re-processing the same audio file
   - Delete this file to force re-transcription

2. **`diarized.txt`**: Speaker-separated transcript
   - Contains the final output with speaker labels (Speaker 1:, Speaker 2:, etc.)
   - Regenerated each time the tool runs

## Configuration

The tool uses the following default settings:

- **Transcription Timeout**: 5 minutes
- **Diarization Timeout**: 2 minutes  
- **HTTP Timeout**: 30 seconds
- **Max Audio File Size**: 25MB
- **Max Response Body Size**: 10MB

These can be modified in the `config` variable in `main.go`.

## Supported Audio Formats

The tool supports any audio format that OpenAI Whisper accepts, including:
- MP3
- WAV
- M4A
- FLAC
- And others

## Error Handling

The tool includes robust error handling for:
- Missing or invalid audio files
- Network timeouts
- API rate limits
- File size limits
- Invalid API responses

## Workflow

1. **File Check**: Checks if `transcription.txt` exists to avoid re-transcription
2. **Audio Transcription**: Uploads audio to Whisper API if no cached transcription
3. **Cache Save**: Saves raw transcription to `transcription.txt`
4. **Speaker Diarization**: Processes transcript through GPT-4 for speaker separation
5. **Output Generation**: Creates `diarized.txt` with labeled speakers

## Troubleshooting

### Common Issues

**"Please set the OPENAI_API_KEY environment variable"**
- Ensure your OpenAI API key is properly set as an environment variable

**"audio file too large"**
- Audio files must be under 25MB. Consider compressing or splitting large files

**"failed to send request" or timeout errors**
- Check your internet connection
- Verify your API key has sufficient credits
- Large files may take longer to process

**"non-200 response" errors**
- Check your API key permissions
- Verify you have access to Whisper and GPT-4 APIs
- Check OpenAI API status

### Performance Tips

- Delete `transcription.txt` only when you want to re-process the audio
- Use compressed audio formats (MP3) for faster uploads
- Ensure stable internet connection for large files

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

### Code Formatting

```bash
go fmt ./...
go vet ./...
```

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]