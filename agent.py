"""PydanticAI agent construction and configuration."""

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from rag_pipeline import RAGPipeline

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using transcript context.

Instructions:
- When context from transcripts is provided, answer based on that context first
- Always cite which transcript the information came from when using retrieved context
- If the context does not contain relevant information, say so clearly, then answer from general knowledge if appropriate
- Maintain continuity across the conversation history provided
- Be concise and clear in your answers"""


def build_agent(model_name: str = "claude-haiku-4-5-20251001") -> Agent:
    """
    Build and configure a PydanticAI agent with RAG support.

    Args:
        model_name: Anthropic model name to use

    Returns:
        Configured Agent instance with transcribe_tool registered
    """
    anthropic_model = AnthropicModel(model_name=model_name)
    agent = Agent(model=anthropic_model, system_prompt=SYSTEM_PROMPT, deps_type=RAGPipeline)

    # Import here to avoid circular imports
    from agent_tools import transcribe_tool
    agent.tool()(transcribe_tool)

    return agent
