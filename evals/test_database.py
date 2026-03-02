"""Unit tests for database.py module."""

import pytest
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import database


class TestDatabaseInit:
    """Tests for database initialization."""

    def test_init_db_idempotent(self, tmp_db):
        """Calling init_db() twice should not raise an error."""
        # First call already happened in the fixture
        # Second call should be safe
        database.init_db()
        database.init_db()
        # If we get here, it was idempotent
        assert True


class TestAddAndRetrieve:
    """Tests for add_transcription and get_transcription_by_id."""

    def test_add_and_retrieve(self, tmp_db):
        """Adding a transcription should return an ID, and retrieving by ID should work."""
        record_id = database.add_transcription(
            input_source="https://youtube.com/watch?v=abc",
            source_type="url",
            transcript_file="/path/to/transcript.txt",
            model_used="base",
            output_format="txt",
            file_size_mb=1.5,
        )

        assert isinstance(record_id, int)
        assert record_id > 0

        # Retrieve the record
        result = database.get_transcription_by_id(record_id)

        assert result is not None
        assert len(result) == 8  # 8 columns in the table
        assert result[1] == "https://youtube.com/watch?v=abc"  # input_source
        assert result[2] == "url"  # source_type

    def test_add_transcription_with_none_file_size(self, tmp_db):
        """Adding transcription with None file_size_mb should work."""
        record_id = database.add_transcription(
            input_source="/local/file.mp3",
            source_type="local_file",
            transcript_file="/path/to/transcript.txt",
            model_used="small",
            output_format="json",
            file_size_mb=None,
        )

        assert isinstance(record_id, int)

        result = database.get_transcription_by_id(record_id)
        assert result is not None


class TestGetNonexistentId:
    """Tests for get_transcription_by_id with invalid IDs."""

    def test_get_nonexistent_id_returns_none(self, tmp_db):
        """Getting a non-existent record ID should return None, not raise."""
        result = database.get_transcription_by_id(9999)

        assert result is None

    def test_get_nonexistent_id_with_empty_db(self, tmp_db):
        """Getting any ID from an empty database should return None."""
        result = database.get_transcription_by_id(1)

        assert result is None


class TestDeleteRecord:
    """Tests for delete_record."""

    def test_delete_existing_record(self, tmp_db):
        """Deleting an existing record should return True."""
        record_id = database.add_transcription(
            input_source="https://example.com",
            source_type="url",
            transcript_file="/path/to/file.txt",
            model_used="base",
            output_format="txt",
            file_size_mb=0.5,
        )

        result = database.delete_record(record_id)

        assert result is True

        # Verify it's deleted
        assert database.get_transcription_by_id(record_id) is None

    def test_delete_nonexistent_record(self, tmp_db):
        """Deleting a non-existent record should return False."""
        result = database.delete_record(9999)

        assert result is False

    def test_delete_same_record_twice(self, tmp_db):
        """Deleting the same record twice should return True then False."""
        record_id = database.add_transcription(
            input_source="https://example.com",
            source_type="url",
            transcript_file="/path/to/file.txt",
            model_used="base",
            output_format="txt",
            file_size_mb=0.5,
        )

        first_delete = database.delete_record(record_id)
        second_delete = database.delete_record(record_id)

        assert first_delete is True
        assert second_delete is False


class TestGetStatistics:
    """Tests for get_statistics."""

    def test_get_statistics_empty_db(self, tmp_db):
        """get_statistics() on an empty database should not crash."""
        stats = database.get_statistics()

        assert isinstance(stats, dict)
        assert "total_transcriptions" in stats
        assert stats["total_transcriptions"] == 0
        assert stats["total_file_size_mb"] == 0.0

    def test_get_statistics_with_records(self, tmp_db):
        """get_statistics() with records should calculate totals correctly."""
        database.add_transcription(
            input_source="url1",
            source_type="url",
            transcript_file="/path/1.txt",
            model_used="base",
            output_format="txt",
            file_size_mb=1.0,
        )

        database.add_transcription(
            input_source="file2",
            source_type="local_file",
            transcript_file="/path/2.txt",
            model_used="small",
            output_format="json",
            file_size_mb=2.5,
        )

        stats = database.get_statistics()

        assert stats["total_transcriptions"] == 2
        assert stats["total_file_size_mb"] == 3.5

    def test_get_statistics_with_none_file_sizes(self, tmp_db):
        """get_statistics() should handle NULL file sizes gracefully."""
        database.add_transcription(
            input_source="url1",
            source_type="url",
            transcript_file="/path/1.txt",
            model_used="base",
            output_format="txt",
            file_size_mb=None,
        )

        stats = database.get_statistics()

        assert stats["total_transcriptions"] == 1
        # NULL values should be treated as 0 in the SUM
        assert stats["total_file_size_mb"] == 0.0


class TestSearchTranscriptions:
    """Tests for search_transcriptions."""

    def test_search_finds_matching_records(self, tmp_db):
        """search_transcriptions should find records matching the search term."""
        database.add_transcription(
            input_source="https://youtube.com/watch?v=abc",
            source_type="url",
            transcript_file="/path/to/transcript.txt",
            model_used="base",
            output_format="txt",
            file_size_mb=1.5,
        )

        database.add_transcription(
            input_source="/local/file.mp3",
            source_type="local_file",
            transcript_file="/path/to/other.txt",
            model_used="small",
            output_format="json",
            file_size_mb=2.0,
        )

        results = database.search_transcriptions("youtube")

        assert len(results) == 1
        assert "youtube" in results[0][1].lower()

    def test_search_returns_empty_for_no_matches(self, tmp_db):
        """search_transcriptions should return empty list for no matches."""
        database.add_transcription(
            input_source="https://example.com",
            source_type="url",
            transcript_file="/path/to/transcript.txt",
            model_used="base",
            output_format="txt",
            file_size_mb=1.5,
        )

        results = database.search_transcriptions("nonexistent")

        assert results == []

    def test_search_empty_database(self, tmp_db):
        """search_transcriptions on empty database should return empty list."""
        results = database.search_transcriptions("anything")

        assert results == []


class TestTranscriptionHistory:
    """Tests for get_transcription_history."""

    def test_get_history_empty_db(self, tmp_db):
        """get_transcription_history on empty DB should return empty list."""
        history = database.get_transcription_history(limit=10)

        assert history == []

    def test_get_history_with_records(self, tmp_db):
        """get_transcription_history should return records in reverse order."""
        database.add_transcription(
            input_source="url1",
            source_type="url",
            transcript_file="/path/1.txt",
            model_used="base",
            output_format="txt",
            file_size_mb=1.0,
        )

        database.add_transcription(
            input_source="url2",
            source_type="url",
            transcript_file="/path/2.txt",
            model_used="base",
            output_format="txt",
            file_size_mb=1.0,
        )

        history = database.get_transcription_history(limit=10)

        assert len(history) == 2
        # Most recent should be first (reverse order)
        assert "url2" in history[0][1]

    def test_get_history_respects_limit(self, tmp_db):
        """get_transcription_history should respect the limit parameter."""
        for i in range(5):
            database.add_transcription(
                input_source=f"url{i}",
                source_type="url",
                transcript_file=f"/path/{i}.txt",
                model_used="base",
                output_format="txt",
                file_size_mb=1.0,
            )

        history = database.get_transcription_history(limit=3)

        assert len(history) == 3
