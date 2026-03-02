"""Main entry point for RAG Chatbot application."""

import sys


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # CLI mode
        from cli import chat_with_anthropic
        model_name = sys.argv[2] if len(sys.argv) > 2 else "claude-haiku-4-5-20251001"
        chat_with_anthropic(model_name)
    else:
        # Gradio UI mode (default)
        from ui import launch_gradio_app
        launch_gradio_app()