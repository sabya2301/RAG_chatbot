"""Tools available to PydanticAI agent."""

from pathlib import Path
from pydantic_ai import RunContext
from rag_pipeline import RAGPipeline
from pipeline import run_transcription
from manifest import save_indexed_manifest, has_new_transcripts


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
        ctx: PydanticAI RunContext with RAGPipeline dependencies
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
