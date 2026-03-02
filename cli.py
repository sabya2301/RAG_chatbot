"""Command-line interface for the RAG chatbot."""

from pathlib import Path
from rag_pipeline import RAGPipeline
from agent import build_agent
from manifest import MANIFEST_FILE, save_indexed_manifest


def chat_with_anthropic(model_name: str = "claude-haiku-4-5-20251001"):
    """
    Interactive CLI chat with RAG-augmented assistant.

    Args:
        model_name: Anthropic model name to use
    """
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

    agent = build_agent(model_name)
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
