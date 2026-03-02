"""
RAG Pipeline with hybrid retrieval using BM25 and Chroma embeddings.
Supports persistent indexing and incremental updates.
"""

import os
from pathlib import Path
from typing import List, Dict
from rank_bm25 import BM25Okapi
import chromadb
from config import TRANSCRIPTS_DIR


class RAGPipeline:
    """RAG pipeline with BM25 and Chroma hybrid retrieval."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize RAG pipeline.

        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Dict[str, str]] = []
        self.bm25_index = None
        self.chroma_client = None
        self.chroma_collection = None
        self.persist_dir = Path(TRANSCRIPTS_DIR).parent / ".chroma_db"

    def load_transcripts(self) -> List[Dict[str, str]]:
        """
        Load all .txt transcripts from TRANSCRIPTS_DIR.

        Returns:
            List of dicts with 'text' and 'source' keys
        """
        documents = []
        transcripts_dir = Path(TRANSCRIPTS_DIR)

        if not transcripts_dir.exists():
            print(f"⚠ Transcripts directory not found: {transcripts_dir}")
            return documents

        txt_files = list(transcripts_dir.glob("*.txt"))
        if not txt_files:
            print(f"⚠ No .txt files found in {transcripts_dir}")
            return documents

        print(f"📄 Loading {len(txt_files)} transcript(s)...")
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    documents.append({
                        'text': text,
                        'source': file_path.name
                    })
                    print(f"   ✓ {file_path.name} ({len(text)} chars)")
            except Exception as e:
                print(f"   ✗ Error reading {file_path.name}: {e}")

        return documents

    def chunk_text(self, text: str, source: str) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            source: Source filename

        Returns:
            List of chunk dicts with 'text', 'source', and 'chunk_id'
        """
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append({
                'text': chunk_text,
                'source': source,
                'chunk_id': f"{source}_{chunk_id}"
            })

            chunk_id += 1
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _init_chroma(self):
        """Initialize Chroma client and collection."""
        self.persist_dir.mkdir(exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )

    def _load_existing_collection(self) -> bool:
        """
        Try to load an existing Chroma collection.

        Returns:
            True if collection loaded successfully, False otherwise
        """
        try:
            self._init_chroma()
            self.chroma_collection = self.chroma_client.get_collection(
                name="transcripts"
            )
            print(f"✓ Loaded existing Chroma collection")
            return True
        except Exception:
            return False

    def build_index(self, force_rebuild: bool = False):
        """
        Build BM25 and Chroma indexes from loaded transcripts.

        Args:
            force_rebuild: If True, force rebuild even if collection exists
        """
        # Load transcripts
        documents = self.load_transcripts()
        if not documents:
            print("❌ No transcripts to index")
            return False

        # Try loading existing collection if not forcing rebuild
        if not force_rebuild:
            print("\n🔍 Checking for existing indexes...")
            if self._load_existing_collection():
                # Reconstruct chunks from collection metadata
                print("   (Note: BM25 index will be rebuilt from transcripts)")
                print("\n✂️  Chunking text...")
                all_chunks = []
                for doc in documents:
                    chunks = self.chunk_text(doc['text'], doc['source'])
                    all_chunks.extend(chunks)
                self.chunks = all_chunks

                # Rebuild BM25 only
                print(f"\n📚 Building BM25 index...")
                texts_tokenized = [chunk['text'].lower().split() for chunk in self.chunks]
                self.bm25_index = BM25Okapi(texts_tokenized)
                print(f"   ✓ BM25 index ready ({len(self.chunks)} chunks)")
                print(f"\n✅ Indexes loaded successfully!")
                return True

        # Full rebuild: chunk and index everything
        print("\n🔨 Building indexes (full rebuild)...")

        # Chunk all documents
        print(f"\n✂️  Chunking text (size={self.chunk_size}, overlap={self.chunk_overlap})...")
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc['text'], doc['source'])
            all_chunks.extend(chunks)
            print(f"   ✓ {doc['source']}: {len(chunks)} chunks")

        self.chunks = all_chunks
        print(f"   📊 Total chunks: {len(self.chunks)}")

        # Build BM25 index
        print(f"\n📚 Building BM25 index...")
        texts_tokenized = [chunk['text'].lower().split() for chunk in self.chunks]
        self.bm25_index = BM25Okapi(texts_tokenized)
        print(f"   ✓ BM25 index ready")

        # Build Chroma collection
        print(f"\n🎨 Building Chroma embeddings...")
        self._init_chroma()

        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(name="transcripts")
        except Exception:
            pass

        # Create new collection
        self.chroma_collection = self.chroma_client.create_collection(
            name="transcripts",
            metadata={"hnsw:space": "cosine"}
        )

        # Add documents to Chroma
        chunk_ids = [chunk['chunk_id'] for chunk in self.chunks]
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        chunk_metadatas = [
            {'source': chunk['source']}
            for chunk in self.chunks
        ]

        self.chroma_collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            metadatas=chunk_metadatas
        )
        print(f"   ✓ Chroma collection ready ({len(self.chunks)} documents)")

        print(f"\n✅ Indexes built successfully!")
        return True

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        Hybrid retrieval using BM25 and Chroma with Reciprocal Rank Fusion.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of retrieved chunks with 'text', 'source', and 'score'
        """
        if not self.chunks or self.bm25_index is None or self.chroma_collection is None:
            return []

        # BM25 retrieval
        query_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        bm25_ranked = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )[:k*2]  # Get more than k for merging
        bm25_results = {idx: score for idx, score in bm25_ranked}

        # Chroma retrieval
        chroma_results = self.chroma_collection.query(
            query_texts=[query],
            n_results=k*2
        )

        # Reciprocal Rank Fusion scoring
        rrf_scores = {}
        k_rrf = 60  # RRF constant

        for rank, (idx, score) in enumerate(bm25_results.items()):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + k_rrf)

        if chroma_results['ids'] and chroma_results['ids'][0]:
            for rank, chunk_id in enumerate(chroma_results['ids'][0]):
                # Find index of this chunk
                chunk_idx = next(
                    (i for i, c in enumerate(self.chunks) if c['chunk_id'] == chunk_id),
                    None
                )
                if chunk_idx is not None:
                    rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1 / (rank + k_rrf)

        # Get top-k results
        top_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        retrieved = []
        seen_sources = set()

        for idx, score in top_results:
            chunk = self.chunks[idx]
            key = (chunk['source'], chunk['text'][:50])

            if key not in seen_sources:
                retrieved.append({
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'score': score
                })
                seen_sources.add(key)

        return retrieved

    def format_context(self, retrieved_chunks: List[Dict[str, str]]) -> str:
        """
        Format retrieved chunks as context for the LLM.

        Args:
            retrieved_chunks: List of retrieved chunks

        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."

        context = "Retrieved context:\n\n"
        for i, chunk in enumerate(retrieved_chunks, 1):
            context += f"[{i}] From {chunk['source']}:\n"
            context += f"{chunk['text']}\n\n"

        return context
