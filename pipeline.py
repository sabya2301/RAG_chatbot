"""Transcription pipeline orchestration."""

from pathlib import Path
from youtube_downloader_simple import download_and_extract_audio
from transcriber import transcribe_audio, save_transcript, is_url, validate_local_file
from database import init_db, add_transcription


def run_transcription(
    input_source: str,
    model: str = "base",
    output_format: str = "txt",
    output_name: str = None
) -> dict:
    """
    Process a single transcription (URL or local file).
    Reusable function for both CLI and agent-based transcription.

    Args:
        input_source: YouTube URL or local file path
        model: Whisper model to use
        output_format: Output format (txt, json, srt, vtt)
        output_name: Custom output filename

    Returns:
        Dict with keys: success, message, record_id, transcript_path, source_type
    """
    try:
        # Initialize database
        init_db()

        # Check if input is URL or local file
        file_size_mb = None
        if is_url(input_source):
            # Handle YouTube URL
            print(f"\n   📥 Downloading from YouTube...")
            media_path = download_and_extract_audio(input_source)
            source_name = Path(media_path).stem
            source_type = "url"
            file_size_mb = Path(media_path).stat().st_size / (1024*1024)
        else:
            # Handle local file
            print(f"\n   📁 Validating local file...")
            file_path = validate_local_file(input_source)
            media_path = str(file_path)
            source_name = file_path.stem
            source_type = "local_file"
            file_size_mb = file_path.stat().st_size / (1024*1024)
            print(f"   ✓ File valid: {file_path.name} ({file_size_mb:.2f} MB)")

        # Transcribe
        print(f"   🎙️  Transcribing with {model} model...")
        result = transcribe_audio(media_path, model_name=model)

        # Save transcript
        print(f"   💾 Saving transcript...")
        final_output_name = output_name or source_name
        transcript_path = save_transcript(result, final_output_name, format_type=output_format)

        # Log to database
        record_id = add_transcription(
            input_source=input_source,
            source_type=source_type,
            transcript_file=transcript_path,
            model_used=model,
            output_format=output_format,
            file_size_mb=file_size_mb
        )

        return {
            'success': True,
            'message': f"✓ Transcribed successfully (ID: {record_id})",
            'record_id': record_id,
            'transcript_path': transcript_path,
            'source_type': source_type
        }

    except FileNotFoundError as e:
        return {
            'success': False,
            'message': f"❌ File Error: {e}",
            'record_id': None,
            'transcript_path': None
        }
    except ValueError as e:
        return {
            'success': False,
            'message': f"❌ Format Error: {e}",
            'record_id': None,
            'transcript_path': None
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"❌ Error: {e}",
            'record_id': None,
            'transcript_path': None
        }
