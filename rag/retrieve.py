# rag/retrieve.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pairs", default="data/metadata/pairs.jsonl")
    ap.add_argument("--faiss_text", default="data/vector_db/faiss_text.index")
    ap.add_argument("--text_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise SystemExit(f"Missing pairs: {pairs_path}")

    pairs = load_jsonl(pairs_path)

    index = faiss.read_index(args.faiss_text)
    if index.ntotal != len(pairs):
        # Not fatal, but warn (ideally they match)
        print(f"[warn] faiss ntotal={index.ntotal} but pairs={len(pairs)}")

    model = SentenceTransformer(args.text_model, device=args.device)
    q = model.encode([args.query], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32)

    scores, ids = index.search(q, args.k)

    print(f"\nQuery: {args.query}\nTop-{args.k} results:\n")
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), 1):
        if idx < 0:
            continue
        r = pairs[idx]
        print(f"{rank:02d}) score={score:.4f}  doc_id={r['doc_id']}")
        print(f"    image: {r['image_path']}")
        print(f"    dicom:  {r['dicom_path']}")
        print(f"    text:   {r['text']}\n")


if __name__ == "__main__":
    main()
