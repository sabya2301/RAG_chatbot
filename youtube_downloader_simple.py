import subprocess
from pathlib import Path
from config import AUDIO_DIR
import re


def download_and_extract_audio(url: str, audio_name: str = None) -> str:
    """
    Download YouTube video and extract audio as MP3 in one step.

    Args:
        url: YouTube video URL
        audio_name: Optional custom name for the audio file

    Returns:
        Path to the extracted audio file
    """
    try:
        # Create output path
        output_template = str(AUDIO_DIR / "%(id)s_%(title)s.%(ext)s")

        # Build yt-dlp command
        cmd = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "mp3",  # Convert to MP3
            "-o", output_template,  # Output template
            url
        ]

        print(f"Downloading and extracting audio from: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Extract video ID from URL to construct expected filename
        video_id_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&\s?]+)', url)
        if not video_id_match:
            raise RuntimeError("Could not extract video ID from URL")

        video_id = video_id_match.group(1)

        # Find the downloaded file starting with this video ID
        audio_files = list(AUDIO_DIR.glob(f"{video_id}_*.mp3"))

        if audio_files:
            audio_file = str(audio_files[0])  # Should only be one with this ID
            print(f"✓ Audio extracted: {audio_file}")
            return audio_file
        else:
            raise RuntimeError(f"Could not locate audio file for video ID: {video_id}")

    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e.stderr}")
        raise
    except FileNotFoundError:
        print("✗ yt-dlp not found. Install it with: pip install yt-dlp")
        raise
    except Exception as e:
        print(f"✗ Error downloading/extracting audio: {e}")
        raise
