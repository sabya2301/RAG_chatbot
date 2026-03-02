"""Shared pytest fixtures and configuration for eval suite."""

import sqlite3
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import sys

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import database
import manifest
import rag_pipeline
import config


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fixture that provides a temporary SQLite database for testing.

    Monkeypatches database.DB_PATH to use a temp file, initializes the DB,
    and returns the path.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    return db_path


@pytest.fixture
def tmp_manifest(tmp_path, monkeypatch):
    """Fixture that provides a temporary directory for manifest testing.

    Monkeypatches manifest.MANIFEST_FILE to a temp file and
    has_new_transcripts() to use a temp transcripts dir.
    Returns the temp path.
    """
    manifest_file = tmp_path / ".indexed_manifest.json"
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    monkeypatch.setattr(manifest, "MANIFEST_FILE", manifest_file)

    # Patch the has_new_transcripts function's Path('transcripts') call
    def mock_has_new_transcripts():
        """Mock implementation that uses the temp transcripts dir."""
        indexed = manifest.get_indexed_files()
        # Handle missing directory gracefully
        if not transcripts_dir.exists():
            return bool(indexed)  # True if there were indexed files, False if none
        current = {f.name for f in transcripts_dir.glob("*.txt")}
        return indexed != current

    monkeypatch.setattr(manifest, "has_new_transcripts", mock_has_new_transcripts)

    return {"tmp_path": tmp_path, "manifest_file": manifest_file, "transcripts_dir": transcripts_dir}


@pytest.fixture
def tmp_rag(tmp_path, monkeypatch):
    """Fixture that provides a RAGPipeline with temporary Chroma directory.

    Creates RAGPipeline with a temp Chroma dir to avoid polluting the real index.
    """
    chroma_dir = tmp_path / ".chroma_db"

    # Create a RAGPipeline instance with custom persist_dir
    # We'll patch the __init__ to use our temp dir
    original_init = rag_pipeline.RAGPipeline.__init__

    def patched_init(self, chunk_size=500, chunk_overlap=50):
        original_init(self, chunk_size, chunk_overlap)
        self.persist_dir = chroma_dir
        # Don't connect to Chroma yet - let the test do that

    monkeypatch.setattr(rag_pipeline.RAGPipeline, "__init__", patched_init)

    rag = rag_pipeline.RAGPipeline()
    rag.persist_dir = chroma_dir  # Ensure it's set

    return rag


@pytest.fixture(scope="session")
def built_rag():
    """Session-scoped fixture that builds RAG index from real transcripts.

    This is expensive to build (embeds transcripts in Chroma), so we do it once
    per test session and reuse it.

    Returns the built RAGPipeline instance.
    """
    rag = rag_pipeline.RAGPipeline()

    # Build index from real transcripts in the project
    transcripts = rag.load_transcripts()

    if transcripts:
        rag.build_index(force_rebuild=False)

    return rag
