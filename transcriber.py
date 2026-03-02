import whisper
from pathlib import Path
from config import WHISPER_MODEL, LANGUAGE, TRANSCRIPTS_DIR, TRANSCRIPT_FORMAT
import json
from urllib.parse import urlparse


def is_url(input_string: str) -> bool:
    """
    Check if input is a URL or a local file path.

    Args:
        input_string: String to check (URL or file path)

    Returns:
        True if it's a URL, False if it's a local file path
    """
    try:
        result = urlparse(input_string)
        return result.scheme in ('http', 'https')
    except Exception:
        return False


def validate_local_file(file_path: str) -> Path:
    """
    Validate that a local file exists and is a supported format.

    Args:
        file_path: Path to the video or audio file

    Returns:
        Path object of the validated file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    file = Path(file_path).expanduser()

    if not file.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    supported_extensions = {
        # Audio formats
        '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg',
        # Video formats
        '.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv', '.wmv', '.m4v'
    }

    if file.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported format: {file.suffix}. "
            f"Supported: {', '.join(sorted(supported_extensions))}"
        )

    return file


def transcribe_audio(audio_path: str, model_name: str = None) -> dict:
    """
    Transcribe audio or video file using OpenAI Whisper.

    Args:
        audio_path: Path to the audio or video file
        model_name: Whisper model to use (tiny, base, small, medium, large)

    Returns:
        Dictionary containing transcription result with text and segments
    """
    try:
        audio_file = Path(audio_path).expanduser()
        if not audio_file.exists():
            raise FileNotFoundError(f"File not found: {audio_path}")

        model = whisper.load_model(model_name or WHISPER_MODEL)
        print(f"Transcribing: {audio_path}")
        print(f"Using model: {model_name or WHISPER_MODEL}")

        result = model.transcribe(str(audio_file), language=LANGUAGE)
        print(f"✓ Transcription complete")
        return result

    except Exception as e:
        print(f"✗ Error transcribing: {e}")
        raise


def save_transcript(result: dict, output_name: str, format_type: str = None) -> str:
    """
    Save transcription result in desired format.

    Args:
        result: Transcription result from whisper
        output_name: Base name for output file (without extension)
        format_type: Format type (txt, json, srt, vtt)

    Returns:
        Path to saved transcript file
    """
    try:
        fmt = format_type or TRANSCRIPT_FORMAT
        output_name = Path(output_name).stem  # Remove any extension
        output_file = TRANSCRIPTS_DIR / f"{output_name}.{fmt}"

        if fmt == "txt":
            with open(output_file, 'w') as f:
                f.write(result['text'])

        elif fmt == "json":
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

        elif fmt == "srt":
            with open(output_file, 'w') as f:
                for i, segment in enumerate(result['segments'], 1):
                    start = _format_timestamp(segment['start'])
                    end = _format_timestamp(segment['end'])
                    text = segment['text'].strip()
                    f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

        elif fmt == "vtt":
            with open(output_file, 'w') as f:
                f.write("WEBVTT\n\n")
                for segment in result['segments']:
                    start = _format_timestamp_vtt(segment['start'])
                    end = _format_timestamp_vtt(segment['end'])
                    text = segment['text'].strip()
                    f.write(f"{start} --> {end}\n{text}\n\n")
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        print(f"✓ Transcript saved: {output_file}")
        return str(output_file)

    except Exception as e:
        print(f"✗ Error saving transcript: {e}")
        raise


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
