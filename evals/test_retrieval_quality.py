"""Integration tests for RAG retrieval quality using real transcripts."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRetrievalQuality:
    """Integration tests for RAG retrieval using real transcripts."""

    def test_retrieves_mcp_chunks_for_mcp_query(self, built_rag):
        """Query about MCP should retrieve chunks from MCP transcript."""
        results = built_rag.retrieve("Model Context Protocol", k=5)

        # Should find results
        assert len(results) > 0

        # At least one result should be from the MCP transcript
        # MCP transcript filename contains "uBL0siiliGo"
        sources = [r["source"] for r in results]

        # Check if any source contains the MCP video ID
        mcp_found = any("uBL0siiliGo" in source for source in sources)

        assert mcp_found, f"MCP transcript not found in results. Sources: {sources}"

    def test_retrieves_data_modeling_chunks(self, built_rag):
        """Query about data modeling should retrieve from data modeling transcript."""
        results = built_rag.retrieve("dimensional data modeling data vault", k=5)

        assert len(results) > 0

        # Data modeling transcript filename contains "WDwNow61JVE"
        sources = [r["source"] for r in results]
        data_modeling_found = any("WDwNow61JVE" in source for source in sources)

        assert data_modeling_found, f"Data modeling transcript not found. Sources: {sources}"

    def test_retrieves_ai_agents_chunks(self, built_rag):
        """Query about AI agents should retrieve from AI agents transcript."""
        results = built_rag.retrieve("AI agent architecture usecases", k=5)

        assert len(results) > 0

        # AI agents transcript filename contains "VDhQFBxIgtI"
        sources = [r["source"] for r in results]
        ai_agents_found = any("VDhQFBxIgtI" in source for source in sources)

        assert ai_agents_found, f"AI agents transcript not found. Sources: {sources}"

    def test_retrieves_ai_concepts_chunks(self, built_rag):
        """Query about AI concepts should retrieve from AI concepts transcript."""
        results = built_rag.retrieve("AI concepts explained artificial intelligence", k=5)

        assert len(results) > 0

        # AI concepts transcript filename contains "OYvlznJ4IZQ"
        sources = [r["source"] for r in results]
        ai_concepts_found = any("OYvlznJ4IZQ" in source for source in sources)

        assert ai_concepts_found, f"AI concepts transcript not found. Sources: {sources}"

    def test_retrieve_returns_at_most_k_results(self, built_rag):
        """retrieve(k=n) should return at most n results."""
        k_values = [1, 3, 5, 10]

        for k in k_values:
            results = built_rag.retrieve("artificial intelligence", k=k)
            assert len(results) <= k, f"Got {len(results)} results but requested k={k}"

    def test_retrieve_returns_results_dict_format(self, built_rag):
        """Each retrieved chunk should have text, source, and score keys."""
        results = built_rag.retrieve("machine learning", k=5)

        if len(results) > 0:
            for result in results:
                assert "text" in result
                assert "source" in result
                assert "score" in result

    def test_retrieve_no_duplicates(self, built_rag):
        """Retrieved results should not contain duplicates."""
        results = built_rag.retrieve("AI intelligence concepts", k=10)

        # Create a set of (source, text[:50]) pairs to check for duplicates
        seen = set()
        duplicates = []

        for result in results:
            key = (result["source"], result["text"][:50])
            if key in seen:
                duplicates.append(key)
            seen.add(key)

        assert len(duplicates) == 0, f"Found duplicate results: {duplicates}"

    def test_broad_query_retrieves_multiple_sources(self, built_rag):
        """Broad query should return chunks from multiple sources."""
        results = built_rag.retrieve("artificial intelligence concepts future", k=10)

        # Should have results
        assert len(results) > 0

        # Should have results from at least 2 different sources
        sources = {r["source"] for r in results}

        assert (
            len(sources) >= 2
        ), f"Expected results from 2+ sources, got from: {sources}"

    def test_retrieve_scores_are_positive(self, built_rag):
        """Retrieved chunk scores should be positive numbers."""
        results = built_rag.retrieve("learning", k=5)

        for result in results:
            assert isinstance(result["score"], (int, float))
            assert result["score"] >= 0, f"Negative score: {result['score']}"

    def test_retrieve_with_no_matching_query(self, built_rag):
        """Query with very specific nonsense should return results (RRF returns something)."""
        # Even with nonsense, RRF fusion might return something due to semantic search
        results = built_rag.retrieve("xyzabc123notaword", k=5)

        # With semantic search, we might still get results
        # Just verify it doesn't crash and returns a list
        assert isinstance(results, list)

    def test_context_formatting_with_retrieved_chunks(self, built_rag):
        """format_context() should properly format retrieved chunks."""
        results = built_rag.retrieve("AI models", k=3)

        formatted = built_rag.format_context(results)

        # Should be a string
        assert isinstance(formatted, str)

        # If we have results, should include numbering and sources
        if len(results) > 0:
            assert "[1]" in formatted

            # Should include at least one source
            sources = [r["source"] for r in results]
            assert any(source in formatted for source in sources)

    def test_retrieve_semantic_vs_keyword_fusion(self, built_rag):
        """RRF fusion should combine both semantic and keyword results."""
        # Query that tests both semantic and keyword matching
        results = built_rag.retrieve("context protocol systems", k=5)

        # Should get results (if index was built)
        # The important thing is that RRF is working
        if len(built_rag.chunks) > 0:
            assert len(results) > 0

    def test_retrieve_result_text_content(self, built_rag):
        """Retrieved chunks should have non-empty text."""
        results = built_rag.retrieve("models learning", k=5)

        for result in results:
            assert len(result["text"]) > 0
            assert isinstance(result["text"], str)

    def test_retrieve_with_different_k_values(self, built_rag):
        """Vary k and verify results scale appropriately."""
        result_k1 = built_rag.retrieve("AI", k=1)
        result_k3 = built_rag.retrieve("AI", k=3)
        result_k5 = built_rag.retrieve("AI", k=5)

        assert len(result_k1) <= len(result_k3)
        assert len(result_k3) <= len(result_k5)
