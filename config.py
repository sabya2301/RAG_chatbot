import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent

# Directories
DOWNLOADS_DIR = PROJECT_ROOT / "downloads" / "videos"
AUDIO_DIR = PROJECT_ROOT / "downloads" / "audio"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"

# Create directories if they don't exist
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Whisper settings
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
LANGUAGE = os.getenv("LANGUAGE", "en")

# yt-dlp settings
YT_DLP_OPTS = {
    'format': 'best[ext=mp4]/best',
    'quiet': False,
    'no_warnings': False,
}

# Transcript format: 'txt', 'json', 'srt', 'vtt'
TRANSCRIPT_FORMAT = os.getenv("TRANSCRIPT_FORMAT", "txt")
