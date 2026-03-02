# YouTube Video Transcriber with RAG Chatbot

A comprehensive AI application that transcribes YouTube videos and local audio files using OpenAI Whisper, then enables intelligent Q&A through a Retrieval-Augmented Generation (RAG) chatbot powered by Claude AI.

## Features

- **Multi-source Transcription**: Transcribe multiple YouTube videos and local audio files (MP3, WAV, etc.)
- **RAG Chatbot**: Ask questions about your transcripts with context-aware responses
- **Persistent Storage**: SQLite database tracks all transcriptions
- **Hybrid Search**: Combines BM25 keyword matching with semantic embeddings
- **Conversation History**: Multi-turn Q&A with full conversation context
- **Web UI**: Gradio interface for easy interaction
- **CLI Interface**: Terminal-based chat for power users
- **Comprehensive Testing**: 81+ unit tests with high coverage

## Quick Start

### Prerequisites
- Python 3.11+
- FFmpeg (required by yt-dlp for audio extraction)
- Anthropic API key (for Claude)

### Dependencies

This project requires several Python packages. All dependencies are listed in `requirements.txt` and include:
- **yt-dlp**: YouTube video downloading
- **openai-whisper**: Speech-to-text transcription
- **pydantic-ai**: AI agent framework
- **chromadb**: Vector database for embeddings
- **rank-bm25**: Keyword-based search
- **sentence-transformers**: Semantic embeddings
- **anthropic**: Claude API client
- **gradio**: Web UI framework
- **pytest**: Testing framework

**Installation is critical**: You must run `pip install -r requirements.txt` in your activated virtual environment before running any part of the application.

### Installation

1. **Clone/setup the project**
```bash
cd capstone_project
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Basic Usage

#### Transcribe a YouTube video:
```bash
python3 main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

#### Transcribe a local file:
```bash
python3 main.py /path/to/audio.mp3
```

#### Chat with transcripts (CLI):
```bash
python3 chatbot.py
```

#### Launch web UI:
```bash
python3 main.py --ui
# Opens Gradio interface at http://localhost:7860
```

## Project Architecture

```
capstone_project/
├── main.py                  # CLI entry point & UI dispatcher
├── pipeline.py              # Transcription orchestration
├── agent.py                 # PydanticAI agent factory
├── agent_tools.py           # Transcribe tool definition
├── cli.py                   # Terminal chat interface
├── ui.py                    # Gradio web interface
├── rag_pipeline.py          # RAG indexing & retrieval (hybrid search)
├── chatbot.py               # Standalone chatbot CLI
├── database.py              # SQLite operations
├── transcriber.py           # Whisper wrapper
├── manifest.py              # Transcript indexing manifest
├── config.py                # Configuration settings
├── requirements.txt         # Python dependencies
├── pytest.ini               # Test configuration
├── evals/                   # Comprehensive test suite
│   ├── conftest.py          # Shared test fixtures
│   ├── test_transcriber.py  # Transcriber tests
│   ├── test_database.py     # Database tests
│   ├── test_manifest.py     # Manifest tests
│   ├── test_rag_unit.py     # RAG unit tests
│   └── test_retrieval_quality.py  # Retrieval accuracy tests
├── transcripts/             # Input transcript files
├── downloads/               # Downloaded videos/audio
└── .chroma_db/              # Persistent vector embeddings
```

## Key Components

### Transcription Pipeline
- **transcriber.py**: Wraps OpenAI Whisper with multiple model sizes (tiny, base, small, medium, large)
- **pipeline.py**: Orchestrates transcription and saves in multiple formats
- **database.py**: Tracks all transcriptions with metadata

### RAG System
- **rag_pipeline.py**: Implements hybrid retrieval using:
  - **BM25**: Fast keyword-based search
  - **Chroma**: Semantic embeddings with sentence-transformers
  - **Reciprocal Rank Fusion**: Combines both search methods
- **Persistent Storage**: Vector embeddings cached in `.chroma_db/`
- **Smart Indexing**: Manifest file prevents re-indexing of unchanged files

### Chatbot
- **PydanticAI Framework**: Structured agent with tool use
- **Claude Model**: Uses `claude-haiku-4-5-20251001` (fast, cost-effective)
- **Context Augmentation**: User queries augmented with relevant transcript chunks
- **Conversation History**: Full multi-turn dialogue support

## Configuration

Edit `config.py` to customize:
- Whisper model size (affects accuracy vs speed)
- Chunk size and overlap (for RAG indexing)
- Retrieval parameters (number of chunks, similarity thresholds)
- Database location

### Chatbot Commands
```
help     - Show available commands
history  - Display conversation history
clear    - Clear conversation history
quit     - Exit chatbot
```

### Web UI
```bash
python3 main.py --ui
```
- **Transcribe Tab**: Upload videos/audio files or enter URLs
- **Chat Tab**: Ask questions about indexed transcripts
- **History Tab**: View past transcriptions and conversations

## Testing

Run the comprehensive test suite:
```bash
pytest evals/ -v           # All tests
pytest evals/test_rag_unit.py -v  # RAG tests only
pytest evals/test_retrieval_quality.py -v  # Retrieval accuracy
```

Test coverage includes:
- **19 tests**: Transcriber (URL detection, format saving)
- **17 tests**: Database (CRUD operations, statistics)
- **13 tests**: Manifest (tracking, deduplication)
- **18 tests**: RAG (chunking, formatting, indexing)
- **14 tests**: Retrieval quality (accuracy with real transcripts)

## Environment Variables

Create `.env` file with:
```
ANTHROPIC_API_KEY=sk-ant-xxx...
WHISPER_MODEL=base          # Options: tiny, base, small, medium, large
TRANSCRIPTS_DIR=transcripts
DATABASE_PATH=transcriptions.db
CHROMA_DB_PATH=.chroma_db
```

## Performance

- **Transcription**: ~1-5 minutes depending on video length and model size
- **Indexing**: <1 second for typical transcripts (reuses cached embeddings)
- **Retrieval**: <100ms for RAG queries
- **Response Generation**: 2-5 seconds per chatbot response

## Architecture Decisions

1. **Hybrid Search**: Combines BM25 (fast, keyword-aware) with semantic embeddings for best coverage
2. **Claude Haiku**: Selected for cost-efficiency without sacrificing quality
3. **SQLite**: Lightweight persistence without external dependencies
4. **PydanticAI**: Type-safe agent framework with built-in tool use
5. **Modular Design**: Each module handles one responsibility for maintainability


### Out of memory with large videos
- Use a smaller Whisper model (e.g., `base` instead of `large`)
- Edit `config.py` and set `WHISPER_MODEL = "base"`

### API rate limits
- Responses are cached after first retrieval
- Adjust `k` parameter in RAG queries to reduce tokens used

## Project Status

- Core functionality: ✅ Complete
- Testing suite: ✅ 81 passing tests
- Web UI: ✅ Fully functional
- CLI: ✅ Fully functional
- Documentation: ✅ Complete

