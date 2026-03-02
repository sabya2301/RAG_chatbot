"""Manifest file management to track indexed transcripts."""

import json
from datetime import datetime
from pathlib import Path

MANIFEST_FILE = Path('.indexed_manifest.json')


def save_indexed_manifest(transcript_files):
    """Save list of indexed transcript files to manifest."""
    manifest = {
        'indexed_files': sorted([str(f.name) for f in transcript_files]),
        'last_updated': datetime.now().isoformat()
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))


def get_indexed_files():
    """Get set of previously indexed transcript filenames."""
    if not MANIFEST_FILE.exists():
        return set()
    try:
        data = json.loads(MANIFEST_FILE.read_text())
        return set(data.get('indexed_files', []))
    except (json.JSONDecodeError, IOError):
        return set()


def has_new_transcripts():
    """Check if there are new transcripts not yet indexed."""
    transcripts_dir = Path('transcripts')
    if not transcripts_dir.exists():
        return False

    current_files = set(f.name for f in transcripts_dir.glob('*.txt'))
    indexed_files = get_indexed_files()
    return current_files != indexed_files
