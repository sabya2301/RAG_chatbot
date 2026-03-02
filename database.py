import sqlite3
from pathlib import Path
from datetime import datetime
from config import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "transcriptions.db"


def init_db():
    """Initialize the SQLite database with transcriptions table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_source TEXT NOT NULL,
            source_type TEXT NOT NULL,
            transcript_file TEXT NOT NULL,
            model_used TEXT NOT NULL,
            output_format TEXT NOT NULL,
            processed_at TIMESTAMP NOT NULL,
            file_size_mb REAL,
            status TEXT DEFAULT 'completed'
        )
    ''')

    conn.commit()
    conn.close()


def add_transcription(
    input_source: str,
    source_type: str,
    transcript_file: str,
    model_used: str,
    output_format: str,
    file_size_mb: float = None
) -> int:
    """
    Add a transcription record to the database.

    Args:
        input_source: YouTube URL or file path
        source_type: 'url' or 'local_file'
        transcript_file: Path to the saved transcript
        model_used: Whisper model used (tiny, base, small, medium, large)
        output_format: Output format (txt, json, srt, vtt)
        file_size_mb: File size in MB (optional)

    Returns:
        ID of the inserted record
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO transcriptions (
            input_source, source_type, transcript_file, model_used,
            output_format, processed_at, file_size_mb
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        input_source,
        source_type,
        transcript_file,
        model_used,
        output_format,
        datetime.now().isoformat(),
        file_size_mb
    ))

    conn.commit()
    record_id = cursor.lastrowid
    conn.close()

    return record_id


def get_transcription_history(limit: int = 10):
    """
    Get transcription history from the database.

    Args:
        limit: Number of records to retrieve (default: 10)

    Returns:
        List of tuples containing transcription records
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, input_source, source_type, transcript_file, model_used,
               output_format, processed_at, file_size_mb
        FROM transcriptions
        ORDER BY processed_at DESC
        LIMIT ?
    ''', (limit,))

    records = cursor.fetchall()
    conn.close()

    return records


def get_transcription_by_id(record_id: int):
    """
    Get a specific transcription record by ID.

    Args:
        record_id: ID of the transcription record

    Returns:
        Tuple containing the record details or None if not found
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, input_source, source_type, transcript_file, model_used,
               output_format, processed_at, file_size_mb
        FROM transcriptions
        WHERE id = ?
    ''', (record_id,))

    record = cursor.fetchone()
    conn.close()

    return record


def search_transcriptions(search_term: str):
    """
    Search transcriptions by input source or transcript file name.

    Args:
        search_term: Search term to find in input_source or transcript_file

    Returns:
        List of matching transcription records
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, input_source, source_type, transcript_file, model_used,
               output_format, processed_at, file_size_mb
        FROM transcriptions
        WHERE input_source LIKE ? OR transcript_file LIKE ?
        ORDER BY processed_at DESC
    ''', (f'%{search_term}%', f'%{search_term}%'))

    records = cursor.fetchall()
    conn.close()

    return records


def get_statistics():
    """
    Get statistics about all transcriptions.

    Returns:
        Dictionary containing statistics
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Total transcriptions
    cursor.execute('SELECT COUNT(*) FROM transcriptions')
    total_count = cursor.fetchone()[0]

    # By source type
    cursor.execute('''
        SELECT source_type, COUNT(*) FROM transcriptions
        GROUP BY source_type
    ''')
    by_source = dict(cursor.fetchall())

    # By model used
    cursor.execute('''
        SELECT model_used, COUNT(*) FROM transcriptions
        GROUP BY model_used
    ''')
    by_model = dict(cursor.fetchall())

    # By output format
    cursor.execute('''
        SELECT output_format, COUNT(*) FROM transcriptions
        GROUP BY output_format
    ''')
    by_format = dict(cursor.fetchall())

    # Total file size
    cursor.execute('SELECT SUM(file_size_mb) FROM transcriptions')
    total_size = cursor.fetchone()[0] or 0

    conn.close()

    return {
        'total_transcriptions': total_count,
        'by_source_type': by_source,
        'by_model': by_model,
        'by_format': by_format,
        'total_file_size_mb': round(total_size, 2)
    }


def delete_record(record_id: int) -> bool:
    """
    Delete a transcription record by ID.

    Args:
        record_id: ID of the record to delete

    Returns:
        True if deleted, False if record not found
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('DELETE FROM transcriptions WHERE id = ?', (record_id,))
    conn.commit()

    deleted = cursor.rowcount > 0
    conn.close()

    return deleted
