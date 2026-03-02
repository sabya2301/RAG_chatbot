import sys
import json
from datetime import datetime
from youtube_downloader_simple import download_and_extract_audio
from transcriber import transcribe_audio, save_transcript, is_url, validate_local_file
from database import init_db, add_transcription
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai import RunContext
from pathlib import Path
from rag_pipeline import RAGPipeline
import gradio as gr

# ---------------------------------
# Manifest Management (Prevent Duplicate Indexing)
# ---------------------------------
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


def run_transcription(
    input_source: str,
    model: str = "base",
    output_format: str = "txt",
    output_name: str = None
) -> dict:
    """
    Process a single transcription (URL or local file).
    Reusable function extracted from old main().

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


def transcribe_tool(
    ctx: RunContext[RAGPipeline],
    inputs: str,
    model: str = "base",
    output_format: str = "txt"
) -> str:
    """
    Tool for chatbot to transcribe multiple URLs/paths.
    Parses multiple inputs separated by spaces or commas.

    Args:
        inputs: Space or comma-separated YouTube URLs or file paths
        model: Whisper model to use
        output_format: Output format

    Returns:
        Summary of transcription results
    """
    # Parse inputs - handle space or comma separated
    print(f"\n📝 Processing input: {inputs}...\n")
    input_list = []
    for sep in [',', ' ']:
        if sep in inputs:
            input_list = [i.strip() for i in inputs.split(sep) if i.strip()]
            break

    if not input_list:
        input_list = [inputs.strip()]
    
    print(f"\n ")

    print(f"\n📝 Processing {len(input_list)} input(s)...\n")

    results = {
        'successful': [],
        'failed': [],
        'total': len(input_list)
    }

    for i, input_source in enumerate(input_list, 1):
        print(f"[{i}/{len(input_list)}] Processing: {input_source}...")

        result = run_transcription(
            input_source,
            model=model,
            output_format=output_format
        )

        if result['success']:
            results['successful'].append({
                'input': input_source,
                'record_id': result['record_id'],
                'path': result['transcript_path']
            })
            print(f"   {result['message']}")
        else:
            results['failed'].append({
                'input': input_source,
                'error': result['message']
            })
            print(f"   {result['message']}")

    # Build summary
    summary = f"\n{'='*60}\n"
    summary += f"📊 Transcription Summary: {len(results['successful'])}/{results['total']} successful\n"
    summary += f"{'='*60}\n"

    if results['successful']:
        summary += "\n✅ Successful:\n"
        for item in results['successful']:
            summary += f"  • {item['input'][:50]}... (ID: {item['record_id']})\n"

    if results['failed']:
        summary += "\n❌ Failed:\n"
        for item in results['failed']:
            summary += f"  • {item['input'][:50]}...\n    {item['error']}\n"

    summary += f"\n{'='*60}"

    # Rebuild RAG index after successful transcriptions
    if results['successful']:
        print("\n🔄 Checking for new transcripts to index...")
        if has_new_transcripts():
            print("   Found new transcripts. Rebuilding RAG index...")
            ctx.deps.build_index(force_rebuild=True)
            # Update manifest with newly indexed files
            all_transcripts = list(Path('transcripts').glob('*.txt'))
            save_indexed_manifest(all_transcripts)
            print("✅ RAG index updated and manifest saved.")
            summary += "\n\n🔄 RAG index rebuilt with new transcripts."
        else:
            print("   No new transcripts found. Index is current.")
            summary += "\n\n✓ No new transcripts to index."

    return summary


def chat_with_anthropic(model_name="claude-haiku-4-5-20251001"):

    # Initialize RAG pipeline
    print("\n🚀 Initializing RAG pipeline...")
    rag = RAGPipeline()

    # Check if this is the first run or if manifest is missing
    if not MANIFEST_FILE.exists():
        print("   First run detected. Building initial index...")
        rag.build_index(force_rebuild=False)
        all_transcripts = list(Path('transcripts').glob('*.txt'))
        save_indexed_manifest(all_transcripts)
        print("   ✓ Initial manifest created.")
    else:
        rag.build_index(force_rebuild=False)

    print("✅ RAG pipeline ready.\n")

    anthropic_model = AnthropicModel(model_name=model_name)
    system_prompt = """You are a helpful assistant that answers questions using transcript context.

Instructions:
- When context from transcripts is provided, answer based on that context first
- Always cite which transcript the information came from when using retrieved context
- If the context does not contain relevant information, say so clearly, then answer from general knowledge if appropriate
- Maintain continuity across the conversation history provided
- Be concise and clear in your answers"""
    agent = Agent(model=anthropic_model, system_prompt=system_prompt, deps_type=RAGPipeline)
    agent.tool()(transcribe_tool)
    messages = []  # Conversation history

    print(f"--- Starting chat with {model_name} ---")
    print("Type 'exit', 'quit', or Ctrl+C to stop.")
    print("Commands: 'history' - show conversation, 'clear' - clear history")
    print("-" * 40)

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "history":
                if not messages:
                    print("No conversation history yet.")
                else:
                    print("\n--- Conversation History ---")
                    for msg in messages:
                        role = msg["role"]
                        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        print(f"{role.capitalize()}: {content}")
                    print("---")
                continue

            if user_input.lower() == "clear":
                messages = []
                print("History cleared.")
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            try:
                # Retrieve RAG context
                retrieved_chunks = rag.retrieve(user_input, k=5)
                rag_context = rag.format_context(retrieved_chunks)

                # Build conversation history
                history_context = ""
                if messages:
                    history_context = "Previous conversation:\n"
                    for msg in messages:
                        history_context += f"{msg['role'].capitalize()}: {msg['content']}\n"
                    history_context += "\n"

                # Augmented query: RAG context + history + current question
                full_prompt = (
                    f"{rag_context}\n\n"
                    f"{history_context}"
                    f"User: {user_input}"
                )

                # Run agent with RAG context and conversation history
                result = agent.run_sync(full_prompt, deps=rag)
                assistant_response = result.output

                # Add messages to history
                messages.append({"role": "user", "content": user_input})
                messages.append({"role": "assistant", "content": assistant_response})

                print("Assistant:", assistant_response)

            except Exception as e:
                print(f"\nError communicating with Anthropic: {e}")
                # Remove the user message if there was an error
                if messages and messages[-1]["role"] == "user":
                    messages.pop()

    except KeyboardInterrupt:
        print("\n\nChat interrupted. Goodbye!")


# Global variables for Gradio
_rag = None
_agent = None
_messages = []

def initialize_gradio():
    """Initialize RAG and agent for Gradio."""
    global _rag, _agent, _messages

    print("\n🚀 Initializing RAG pipeline...")
    _rag = RAGPipeline()

    if not MANIFEST_FILE.exists():
        print("   First run detected. Building initial index...")
        _rag.build_index(force_rebuild=False)
        all_transcripts = list(Path('transcripts').glob('*.txt'))
        save_indexed_manifest(all_transcripts)
        print("   ✓ Initial manifest created.")
    else:
        _rag.build_index(force_rebuild=False)

    print("✅ RAG pipeline ready.\n")

    anthropic_model = AnthropicModel(model_name="claude-haiku-4-5-20251001")
    system_prompt = """You are a helpful assistant that answers questions using transcript context.

Instructions:
- When context from transcripts is provided, answer based on that context first
- Always cite which transcript the information came from when using retrieved context
- If the context does not contain relevant information, say so clearly, then answer from general knowledge if appropriate
- Maintain continuity across the conversation history provided
- Be concise and clear in your answers"""
    _agent = Agent(model=anthropic_model, system_prompt=system_prompt, deps_type=RAGPipeline)
    _agent.tool()(transcribe_tool)
    _messages = []

def chat(user_input):
    """Handle a single chat turn."""
    global _rag, _agent, _messages

    if _rag is None or _agent is None:
        initialize_gradio()

    if not user_input.strip():
        return ""

    try:
        # Retrieve RAG context
        retrieved_chunks = _rag.retrieve(user_input, k=5)
        rag_context = _rag.format_context(retrieved_chunks)

        # Build conversation history
        history_context = ""
        if _messages:
            history_context = "Previous conversation:\n"
            for msg in _messages:
                history_context += f"{msg['role'].capitalize()}: {msg['content']}\n"
            history_context += "\n"

        # Augmented query
        full_prompt = (
            f"{rag_context}\n\n"
            f"{history_context}"
            f"User: {user_input}"
        )

        # Run agent
        result = _agent.run_sync(full_prompt, deps=_rag)
        assistant_response = result.output

        # Add to history
        _messages.append({"role": "user", "content": user_input})
        _messages.append({"role": "assistant", "content": assistant_response})

        return assistant_response

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # CLI mode
        model_name = sys.argv[2] if len(sys.argv) > 2 else "claude-haiku-4-5-20251001"
        chat_with_anthropic(model_name)
    else:
        # Gradio UI mode (default)
        initialize_gradio()
        force_dark_mode = """
            function refresh() {
                const url = new URL(window.location);
                if (url.searchParams.get('__theme') !== 'dark') {
                    url.searchParams.set('__theme', 'dark');
                    window.location.href = url.href;
                }
            }
        """
        gr.Interface(
            fn=chat,
            inputs="textbox",
            outputs="textbox",
            title="RAG Chatbot",
            description="Ask questions about your transcripts",
            theme="dark",
            flagging_mode="never"
        ).launch()