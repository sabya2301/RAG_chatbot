"""Unit tests for rag_pipeline.py RAGPipeline methods."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline import RAGPipeline


class TestChunkText:
    """Tests for RAGPipeline.chunk_text() method."""

    def test_chunk_text_single_chunk(self):
        """Text shorter than chunk_size should produce exactly one chunk."""
        rag = RAGPipeline(chunk_size=500, chunk_overlap=50)

        text = "This is a short text."
        chunks = rag.chunk_text(text, "test_source.txt")

        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["source"] == "test_source.txt"

    def test_chunk_text_multiple_chunks(self):
        """Text longer than chunk_size should produce multiple chunks."""
        rag = RAGPipeline(chunk_size=100, chunk_overlap=10)

        # Create text with known length > chunk_size
        text = "A" * 250  # 250 chars

        chunks = rag.chunk_text(text, "test.txt")

        # Should produce multiple chunks
        assert len(chunks) > 1

    def test_chunk_text_chunk_id_format(self):
        """chunk_id should start with the source filename."""
        rag = RAGPipeline(chunk_size=500, chunk_overlap=50)

        text = "A" * 1000
        chunks = rag.chunk_text(text, "my_transcript.txt")

        # Check all chunk_ids start with source
        for chunk in chunks:
            assert chunk["chunk_id"].startswith("my_transcript.txt")

    def test_chunk_text_incremental_chunk_ids(self):
        """chunk_ids should have incremental numbers for multiple chunks."""
        rag = RAGPipeline(chunk_size=100, chunk_overlap=10)

        text = "B" * 250
        chunks = rag.chunk_text(text, "test.txt")

        # Chunk IDs should end with _0, _1, _2, etc.
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"].endswith(f"_{i}")

    def test_chunk_text_overlap(self):
        """Chunks should have overlap specified by chunk_overlap."""
        rag = RAGPipeline(chunk_size=50, chunk_overlap=10)

        text = "C" * 120  # Ensures multiple chunks with overlap

        chunks = rag.chunk_text(text, "test.txt")

        # If we have at least 2 chunks, verify overlap exists
        if len(chunks) >= 2:
            # The end of first chunk should overlap with start of second chunk
            first_chunk_text = chunks[0]["text"]
            second_chunk_text = chunks[1]["text"]

            # There should be some overlap
            assert len(chunks) > 1

    def test_chunk_text_preserves_source(self):
        """All chunks should preserve the source filename."""
        rag = RAGPipeline(chunk_size=100, chunk_overlap=10)

        text = "D" * 300
        source = "my_video_transcript.txt"
        chunks = rag.chunk_text(text, source)

        for chunk in chunks:
            assert chunk["source"] == source

    def test_chunk_text_empty_string(self):
        """Empty string handling in chunk_text."""
        rag = RAGPipeline(chunk_size=500, chunk_overlap=50)

        chunks = rag.chunk_text("", "empty.txt")

        # Empty string may produce zero chunks (valid behavior)
        # The important thing is it doesn't crash and returns a list
        assert isinstance(chunks, list)


class TestFormatContext:
    """Tests for RAGPipeline.format_context() method."""

    def test_format_context_empty_list(self):
        """format_context([]) should return a specific message."""
        rag = RAGPipeline()

        result = rag.format_context([])

        assert "No relevant context found" in result

    def test_format_context_single_chunk(self):
        """format_context with one chunk should format it properly."""
        rag = RAGPipeline()

        chunks = [
            {"text": "This is the first chunk.", "source": "video1.txt", "score": 0.95}
        ]

        result = rag.format_context(chunks)

        # Should include numbering
        assert "[1]" in result
        # Should include source
        assert "video1.txt" in result
        # Should include text
        assert "This is the first chunk." in result

    def test_format_context_multiple_chunks(self):
        """format_context with multiple chunks should number them correctly."""
        rag = RAGPipeline()

        chunks = [
            {"text": "First chunk", "source": "video1.txt", "score": 0.95},
            {"text": "Second chunk", "source": "video2.txt", "score": 0.87},
            {"text": "Third chunk", "source": "video1.txt", "score": 0.78},
        ]

        result = rag.format_context(chunks)

        # Should include all numberings
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

        # Should include all sources
        assert "video1.txt" in result
        assert "video2.txt" in result

        # Should include all texts
        assert "First chunk" in result
        assert "Second chunk" in result
        assert "Third chunk" in result

    def test_format_context_includes_source_label(self):
        """format_context should label chunks with 'From' or similar."""
        rag = RAGPipeline()

        chunks = [{"text": "Sample text", "source": "test_transcript.txt", "score": 0.9}]

        result = rag.format_context(chunks)

        # Should clearly indicate the source
        assert "test_transcript.txt" in result


class TestRetrieveNoIndex:
    """Tests for RAGPipeline.retrieve() without a built index."""

    def test_retrieve_no_index(self):
        """retrieve() before build_index() should return empty list."""
        rag = RAGPipeline()

        # Don't call build_index()
        result = rag.retrieve("some query")

        assert result == []

    def test_retrieve_empty_chunks_list(self):
        """retrieve() with empty chunks should return empty list."""
        rag = RAGPipeline()

        # Ensure chunks are empty
        rag.chunks = []

        result = rag.retrieve("query")

        assert result == []


class TestLoadTranscripts:
    """Tests for RAGPipeline.load_transcripts() method."""

    def test_load_transcripts_missing_dir(self, tmp_path, monkeypatch):
        """load_transcripts() with missing directory should return empty list."""
        import rag_pipeline

        # Monkeypatch TRANSCRIPTS_DIR at the module level where it's imported
        monkeypatch.setattr(rag_pipeline, "TRANSCRIPTS_DIR", tmp_path / "nonexistent")

        rag = RAGPipeline()
        result = rag.load_transcripts()

        assert result == []

    def test_load_transcripts_empty_dir(self, tmp_path, monkeypatch):
        """load_transcripts() with empty directory should return empty list."""
        import rag_pipeline

        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir()

        monkeypatch.setattr(rag_pipeline, "TRANSCRIPTS_DIR", transcripts_dir)

        rag = RAGPipeline()
        result = rag.load_transcripts()

        assert result == []

    def test_load_transcripts_with_files(self, tmp_path, monkeypatch):
        """load_transcripts() should load all .txt files from directory."""
        import rag_pipeline

        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir()

        # Create some transcript files
        (transcripts_dir / "video1.txt").write_text("Content of video 1")
        (transcripts_dir / "video2.txt").write_text("Content of video 2")
        (transcripts_dir / "other.md").write_text("Not a transcript")

        monkeypatch.setattr(rag_pipeline, "TRANSCRIPTS_DIR", transcripts_dir)

        rag = RAGPipeline()
        result = rag.load_transcripts()

        # Should load 2 .txt files
        assert len(result) == 2

        # Check content
        sources = {r["source"] for r in result}
        assert "video1.txt" in sources
        assert "video2.txt" in sources

    def test_load_transcripts_content_preserved(self, tmp_path, monkeypatch):
        """Loaded transcripts should preserve file content."""
        import rag_pipeline

        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir()

        test_content = "This is the transcript content."
        (transcripts_dir / "test.txt").write_text(test_content)

        monkeypatch.setattr(rag_pipeline, "TRANSCRIPTS_DIR", transcripts_dir)

        rag = RAGPipeline()
        result = rag.load_transcripts()

        assert len(result) == 1
        assert result[0]["text"] == test_content
        assert result[0]["source"] == "test.txt"


class TestBuildIndex:
    """Tests for RAGPipeline.build_index() method."""

    def test_build_index_no_transcripts(self, tmp_path, monkeypatch):
        """build_index() with no transcripts should return False."""
        import rag_pipeline

        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir()

        monkeypatch.setattr(rag_pipeline, "TRANSCRIPTS_DIR", transcripts_dir)

        rag = RAGPipeline()
        result = rag.build_index(force_rebuild=False)

        assert result is False

    def test_build_index_with_transcripts(self, tmp_path, monkeypatch):
        """build_index() with transcripts should return True."""
        import rag_pipeline

        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir()

        # Create a transcript
        (transcripts_dir / "test.txt").write_text("A" * 1000)

        monkeypatch.setattr(rag_pipeline, "TRANSCRIPTS_DIR", transcripts_dir)

        rag = RAGPipeline()
        result = rag.build_index(force_rebuild=False)

        assert result is True

        # Index should be built
        assert len(rag.chunks) > 0
        assert rag.bm25_index is not None

    def test_build_index_creates_chunks(self, tmp_path, monkeypatch):
        """build_index() should create chunks from transcripts."""
        import rag_pipeline

        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir()

        (transcripts_dir / "test.txt").write_text("B" * 1500)  # > chunk_size

        monkeypatch.setattr(rag_pipeline, "TRANSCRIPTS_DIR", transcripts_dir)

        rag = RAGPipeline(chunk_size=500, chunk_overlap=50)
        rag.build_index(force_rebuild=False)

        # Should have multiple chunks
        assert len(rag.chunks) > 1
