"""Gradio web interface for the RAG chatbot."""

from pathlib import Path
import gradio as gr
from rag_pipeline import RAGPipeline
from agent import build_agent
from manifest import MANIFEST_FILE, save_indexed_manifest

# Global state for Gradio app
_rag = None
_agent = None
_messages = []


def initialize_gradio():
    """Initialize RAG pipeline and agent for Gradio."""
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
    _agent = build_agent()
    _messages = []


def chat(user_input: str) -> str:
    """
    Handle a single chat turn in Gradio.

    Args:
        user_input: User's message

    Returns:
        Assistant's response
    """
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


def launch_gradio_app():
    """Launch the Gradio web interface."""
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
        js=force_dark_mode,
        flagging_mode="never"
    ).launch(share=True)
