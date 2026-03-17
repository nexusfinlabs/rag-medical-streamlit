# rag/retrieve_numpy.py
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pairs", default="data/metadata/pairs.jsonl")
    ap.add_argument("--text_emb", default="data/embeddings/text_embeddings.npz")
    ap.add_argument("--text_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    emb_path = Path(args.text_emb)
    if not pairs_path.exists():
        raise SystemExit(f"Missing pairs: {pairs_path}")
    if not emb_path.exists():
        raise SystemExit(f"Missing embeddings: {emb_path}")

    pairs = list(load_jsonl(pairs_path))
    E = np.load(emb_path)["embeddings"].astype(np.float32)

    if len(pairs) != E.shape[0]:
        raise SystemExit(f"pairs ({len(pairs)}) != embeddings rows ({E.shape[0]})")

    model = SentenceTransformer(args.text_model, device=args.device)
    q = model.encode([args.query], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32)[0]

    # cosine sim = dot product (porque embeddings están normalizados)
    scores = E @ q
    topk = np.argpartition(-scores, args.k)[: args.k]
    topk = topk[np.argsort(-scores[topk])]

    print(f"\nQuery: {args.query}\nTop-{args.k} results:\n")
    for rank, idx in enumerate(topk, 1):
        r = pairs[int(idx)]
        print(f"{rank:02d}) score={float(scores[idx]):.4f}  doc_id={r['doc_id']}")
        print(f"    image: {r['image_path']}")
        print(f"    dicom:  {r['dicom_path']}")
        print(f"    text:   {r['text']}\n")


if __name__ == "__main__":
    main()
