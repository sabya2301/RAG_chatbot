"""Unit tests for manifest.py module."""

import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import manifest


class TestSaveAndGetRoundtrip:
    """Tests for save_indexed_manifest and get_indexed_files round-trip."""

    def test_save_and_get_roundtrip(self, tmp_manifest):
        """Saving and retrieving manifest should preserve the file list."""
        tmp_dir = tmp_manifest["tmp_path"]
        manifest_file = tmp_manifest["manifest_file"]

        files = [
            Path("transcript1.txt"),
            Path("transcript2.txt"),
            Path("transcript3.txt"),
        ]

        # Save
        manifest.save_indexed_manifest(files)

        # Get
        retrieved = manifest.get_indexed_files()

        # Should match
        assert retrieved == {"transcript1.txt", "transcript2.txt", "transcript3.txt"}

    def test_save_empty_list(self, tmp_manifest):
        """Saving an empty list should create a manifest with empty indexed_files."""
        manifest.save_indexed_manifest([])

        retrieved = manifest.get_indexed_files()

        assert retrieved == set()

    def test_save_preserves_only_filenames(self, tmp_manifest):
        """Manifest should only save filenames, not full paths."""
        files = [
            Path("/some/path/transcript1.txt"),
            Path("../relative/transcript2.txt"),
        ]

        manifest.save_indexed_manifest(files)

        retrieved = manifest.get_indexed_files()

        # Should only have filenames
        assert retrieved == {"transcript1.txt", "transcript2.txt"}


class TestGetIndexedFilesMissingFile:
    """Tests for get_indexed_files with missing manifest file."""

    def test_get_indexed_files_missing_file(self, tmp_manifest):
        """get_indexed_files() should return empty set if manifest doesn't exist."""
        # Don't save anything first
        retrieved = manifest.get_indexed_files()

        assert retrieved == set()

    def test_get_indexed_files_missing_file_empty_dict_returned(self, tmp_manifest):
        """Multiple calls without manifest should all return empty set."""
        result1 = manifest.get_indexed_files()
        result2 = manifest.get_indexed_files()

        assert result1 == set()
        assert result2 == set()


class TestGetIndexedFilesCorruptJson:
    """Tests for get_indexed_files with corrupted manifest file."""

    def test_get_indexed_files_corrupt_json(self, tmp_manifest):
        """get_indexed_files() should return empty set if JSON is corrupt."""
        manifest_file = tmp_manifest["manifest_file"]

        # Write junk to the manifest file
        with open(manifest_file, "w") as f:
            f.write("{ this is not valid json ]}")

        retrieved = manifest.get_indexed_files()

        # Should return empty set, not crash
        assert retrieved == set()

    def test_get_indexed_files_missing_indexed_files_key(self, tmp_manifest):
        """get_indexed_files() should handle JSON missing 'indexed_files' key."""
        manifest_file = tmp_manifest["manifest_file"]

        # Write valid JSON but without the expected key
        with open(manifest_file, "w") as f:
            json.dump({"wrong_key": ["file.txt"]}, f)

        retrieved = manifest.get_indexed_files()

        # Should return empty set
        assert retrieved == set()


class TestHasNewTranscripts:
    """Tests for has_new_transcripts function."""

    def test_has_new_transcripts_no_new_files(self, tmp_manifest):
        """has_new_transcripts() should return False when files haven't changed."""
        transcripts_dir = tmp_manifest["transcripts_dir"]

        # Create some transcript files
        (transcripts_dir / "transcript1.txt").touch()
        (transcripts_dir / "transcript2.txt").touch()

        # Save manifest matching these files
        manifest.save_indexed_manifest(
            [Path("transcript1.txt"), Path("transcript2.txt")]
        )

        # No new transcripts
        assert manifest.has_new_transcripts() is False

    def test_has_new_transcripts_added_file(self, tmp_manifest):
        """has_new_transcripts() should return True when a file is added."""
        transcripts_dir = tmp_manifest["transcripts_dir"]

        # Create initial transcript files
        (transcripts_dir / "transcript1.txt").touch()
        (transcripts_dir / "transcript2.txt").touch()

        # Save manifest with just these two
        manifest.save_indexed_manifest(
            [Path("transcript1.txt"), Path("transcript2.txt")]
        )

        # Now add a new file
        (transcripts_dir / "transcript3.txt").touch()

        # Should detect new file
        assert manifest.has_new_transcripts() is True

    def test_has_new_transcripts_deleted_file(self, tmp_manifest):
        """has_new_transcripts() should return True when a file is deleted."""
        transcripts_dir = tmp_manifest["transcripts_dir"]

        # Create transcript files
        file1 = transcripts_dir / "transcript1.txt"
        file2 = transcripts_dir / "transcript2.txt"
        file1.touch()
        file2.touch()

        # Save manifest with both files
        manifest.save_indexed_manifest([Path("transcript1.txt"), Path("transcript2.txt")])

        # Delete one file
        file2.unlink()

        # Should detect deletion
        assert manifest.has_new_transcripts() is True

    def test_has_new_transcripts_missing_dir(self, tmp_manifest):
        """has_new_transcripts() should handle missing transcripts directory gracefully."""
        transcripts_dir = tmp_manifest["transcripts_dir"]

        # Create and save initial manifest
        (transcripts_dir / "transcript1.txt").touch()
        manifest.save_indexed_manifest([Path("transcript1.txt")])

        # Remove the transcripts directory
        import shutil
        shutil.rmtree(transcripts_dir)

        # Should return True (dir is gone but manifest still has files, so they differ)
        # The function should handle this gracefully without crashing
        result = manifest.has_new_transcripts()
        # When dir is deleted but manifest has files, should indicate a difference
        assert result is True

    def test_has_new_transcripts_empty_dir_with_manifest(self, tmp_manifest):
        """has_new_transcripts() should return True if dir is empty but manifest has files."""
        transcripts_dir = tmp_manifest["transcripts_dir"]

        # Save manifest with files that don't exist
        manifest.save_indexed_manifest([Path("nonexistent.txt")])

        # Dir is empty (no txt files)
        assert manifest.has_new_transcripts() is True

    def test_has_new_transcripts_empty_dir_empty_manifest(self, tmp_manifest):
        """has_new_transcripts() should return False if both dir and manifest are empty."""
        # Don't save any manifest
        # Dir is empty (transcripts_dir exists but has no files)

        # Should return False (no new transcripts, both empty)
        assert manifest.has_new_transcripts() is False
