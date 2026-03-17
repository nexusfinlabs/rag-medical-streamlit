# rag/hybrid_search.py
"""
Hybrid Search: BM25 (keyword) + Vector (semantic) + RRF Fusion + Reranker

WHY HYBRID?
-----------
Vector search alone FAILS for exact terms. Example:
  Query: "NER1006 SIEMENS 3.0mm"
  - Vector search: finds semantically similar but may miss exact IDs
  - BM25: finds exact keyword matches → NER1006 trials, SIEMENS scans, 3.0mm slices
  - Hybrid: gets the best of both worlds

RRF (Reciprocal Rank Fusion):
  score(doc) = Σ 1/(k + rank_in_method)   where k=60
  This normalizes and merges rankings from different search methods.

Usage:
  python -m rag.hybrid_search --query "lung nodule thin slice SIEMENS" --k 10
  python -m rag.hybrid_search --query "NER1006" --k 5 --mode bm25
  python -m rag.hybrid_search --query "chest CT 1.25mm" --k 8 --mode hybrid --rerank
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ─── Data loading ───────────────────────────────────────────
def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ─── BM25 Search ────────────────────────────────────────────
# WHY BM25?
# BM25 = Best Match 25. Used by Elasticsearch, Google (historically).
# Essential for exact term matching:
#   - Medical codes: "NER1006", "NRL972", "QTc"
#   - Scanner IDs: "SIEMENS", "GE MEDICAL SYSTEMS"
#   - Protocol numbers: "1.25mm", "kVp=120"
#   - Patient IDs: exact matches only
#
# Vector search would find "semantically similar" but miss exact terms.

class BM25Index:
    """BM25 index over document texts."""

    def __init__(self, documents: List[str]):
        self.documents = documents
        # Tokenize: lowercase + split on whitespace and common delimiters
        self.tokenized = [self._tokenize(doc) for doc in documents]
        self.index = BM25Okapi(self.tokenized)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenizer — split on spaces, pipes, equals signs."""
        import re
        # Normalize: lowercase, split on non-alphanumeric
        tokens = re.findall(r'[a-z0-9_.]+', text.lower())
        return tokens

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Returns: [(doc_index, bm25_score), ...] sorted by score descending.
        """
        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)
        # Get top-k indices
        if len(scores) <= k:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


# ─── Vector Search ──────────────────────────────────────────
# WHY VECTOR SEARCH?
# Finds documents by MEANING, not keywords.
# "thin slice lung scan" → finds "AX 1.25 LUNG WC" (semantically similar)
# Uses cosine similarity via normalized dot product.

class VectorIndex:
    """Vector search using pre-computed embeddings + SentenceTransformer."""

    def __init__(self, embeddings_path: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        data = np.load(embeddings_path)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.model = SentenceTransformer(model_name, device=device)

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Returns: [(doc_index, cosine_score), ...] sorted by score descending.
        """
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)[0]
        scores = self.embeddings @ q  # dot product = cosine (normalized)
        if len(scores) <= k:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(i), float(scores[i])) for i in top_indices[:k]]


# ─── Reciprocal Rank Fusion (RRF) ──────────────────────────
# WHY RRF?
# Different search methods return different score scales.
# BM25 scores: 0-30+, Vector scores: 0-1.
# You can't just add them. RRF normalizes by RANK:
#   score(doc) = 1/(k + rank_bm25) + 1/(k + rank_vector)
# k=60 is standard (from the original paper).

def reciprocal_rank_fusion(
    *result_lists: List[Tuple[int, float]],
    k: int = 60,
    top_n: int = 10,
) -> List[Tuple[int, float]]:
    """
    Merge multiple ranked lists using RRF.
    
    Args:
        *result_lists: each is [(doc_index, score), ...] sorted by score desc
        k: RRF constant (default 60, from original paper)
        top_n: how many final results to return
    
    Returns:
        [(doc_index, rrf_score), ...] sorted by rrf_score descending
    """
    scores: Dict[int, float] = {}
    for results in result_lists:
        for rank, (doc_idx, _) in enumerate(results, 1):
            if doc_idx not in scores:
                scores[doc_idx] = 0.0
            scores[doc_idx] += 1.0 / (k + rank)
    
    # Sort by RRF score descending
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_n]


# ─── Reranker (HuggingFace Cross-Encoder) ───────────────────
# WHY RERANK?
# Initial retrieval (BM25/vectors) is fast but approximate.
# A cross-encoder is SLOWER but MORE ACCURATE:
#   - It sees query AND document TOGETHER (not separately)
#   - Makes a joint relevance judgment
#   - Typically improves precision by 10-20%
# Use it on the small fused set (top 20→10), not on entire corpus.

def rerank(
    query: str,
    doc_indices: List[int],
    documents: List[str],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 10,
) -> List[Tuple[int, float]]:
    """
    Rerank documents using a HuggingFace cross-encoder.
    
    The cross-encoder scores EACH (query, document) pair for relevance.
    Much more accurate than bi-encoder/BM25 but too slow for full corpus.
    """
    from sentence_transformers import CrossEncoder
    
    reranker = CrossEncoder(model_name)
    pairs = [(query, documents[idx]) for idx in doc_indices if idx < len(documents)]
    valid_indices = [idx for idx in doc_indices if idx < len(documents)]
    
    if not pairs:
        return []
    
    scores = reranker.predict(pairs)
    ranked = sorted(zip(valid_indices, scores), key=lambda x: x[1], reverse=True)
    return [(idx, float(score)) for idx, score in ranked[:top_n]]


# ─── Main Hybrid Search ────────────────────────────────────
class HybridSearchEngine:
    """
    Complete Hybrid Search pipeline:
    1. BM25 keyword search
    2. Vector semantic search
    3. RRF fusion
    4. Optional cross-encoder reranking
    """

    def __init__(
        self,
        pairs_path: str = "data/metadata/pairs.jsonl",
        text_emb_path: str = "data/embeddings/text_embeddings.npz",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        print("Loading data...")
        self.pairs = load_jsonl(Path(pairs_path))
        self.documents = [p.get("text", "") for p in self.pairs]
        
        print(f"Building BM25 index ({len(self.documents):,} docs)...")
        self.bm25 = BM25Index(self.documents)
        
        print("Loading vector index...")
        self.vectors = VectorIndex(Path(text_emb_path), model_name, device)
        
        print(f"✅ Hybrid engine ready: {len(self.pairs):,} documents")

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "hybrid",  # "hybrid", "bm25", "vector"
        use_reranker: bool = False,
        rrf_k: int = 60,
    ) -> List[Dict]:
        """
        Execute search and return results with metadata.
        
        Args:
            query: search query
            k: number of results
            mode: "hybrid" (BM25+Vector+RRF), "bm25" (keyword only), "vector" (semantic only)
            use_reranker: apply cross-encoder reranking
            rrf_k: RRF constant
        """
        t0 = time.time()
        
        if mode == "bm25":
            results = self.bm25.search(query, k=k)
            method = "BM25"
        elif mode == "vector":
            results = self.vectors.search(query, k=k)
            method = "Vector"
        else:  # hybrid
            bm25_results = self.bm25.search(query, k=k * 2)
            vector_results = self.vectors.search(query, k=k * 2)
            results = reciprocal_rank_fusion(bm25_results, vector_results, k=rrf_k, top_n=k * 2)
            method = "Hybrid (BM25 + Vector + RRF)"
        
        if use_reranker and results:
            doc_indices = [idx for idx, _ in results]
            results = rerank(query, doc_indices, self.documents, top_n=k)
            method += " + Reranker"
        
        results = results[:k]
        elapsed = time.time() - t0
        
        # Build output with metadata
        output = []
        for rank, (idx, score) in enumerate(results, 1):
            if idx >= len(self.pairs):
                continue
            pair = self.pairs[idx]
            output.append({
                "rank": rank,
                "score": round(score, 4),
                "doc_id": pair.get("doc_id", ""),
                "text": pair.get("text", ""),
                "image_path": pair.get("image_path", ""),
                "dicom_path": pair.get("dicom_path", ""),
            })
        
        print(f"\n🔍 Query: \"{query}\"")
        print(f"📊 Method: {method}")
        print(f"⏱️  Time: {elapsed:.3f}s")
        print(f"📄 Results: {len(output)}\n")
        
        return output


def main():
    ap = argparse.ArgumentParser(description="Hybrid Search: BM25 + Vector + RRF + Reranker")
    ap.add_argument("--query", required=True, help="Search query")
    ap.add_argument("--k", type=int, default=10, help="Number of results")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "vector"])
    ap.add_argument("--rerank", action="store_true", help="Apply cross-encoder reranking")
    ap.add_argument("--pairs", default="data/metadata/pairs.jsonl")
    ap.add_argument("--text_emb", default="data/embeddings/text_embeddings.npz")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = ap.parse_args()

    engine = HybridSearchEngine(
        pairs_path=args.pairs,
        text_emb_path=args.text_emb,
        device=args.device,
    )

    results = engine.search(args.query, k=args.k, mode=args.mode, use_reranker=args.rerank)

    for r in results:
        print(f"  {r['rank']:02d}) score={r['score']:.4f}  doc_id={r['doc_id']}")
        print(f"      text: {r['text'][:100]}")
        print(f"      image: {r['image_path']}")
        print()


if __name__ == "__main__":
    main()
