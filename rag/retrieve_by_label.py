#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--must_have", default="")  # e.g. "thin_slice,lung_window_like"
    ap.add_argument("--pairs", default="data/metadata/pairs_labeled.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    must = [x.strip() for x in args.must_have.split(",") if x.strip()]
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise SystemExit(f"Missing {pairs_path}")

    rows, texts = [], []
    for line in pairs_path.open():
        r = json.loads(line)
        labels = set(r.get("labels", []))
        if must and not all(m in labels for m in must):
            continue
        rows.append(r)
        texts.append(r["text"])

    if not rows:
        raise SystemExit(f"No rows matched must_have={must}")

    model = SentenceTransformer(args.model, device=args.device)
    X = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    q = model.encode([args.query], normalize_embeddings=True)[0]

    scores = X @ q
    idx = np.argsort(-scores)[:min(args.k, len(scores))]

    print(f"\nQuery: {args.query}")
    print(f"Filter must_have: {must}")
    print(f"Candidates: {len(rows)}")
    print(f"Top-{len(idx)} results:\n")

    for rank, i in enumerate(idx, 1):
        r = rows[int(i)]
        print(f"{rank:02d}) score={float(scores[i]):.4f}")
        print(f"    image: {r['image_path']}")
        print(f"    dicom:  {r['dicom_path']}")
        print(f"    labels: {r.get('labels',[])}")
        print(f"    text:   {r['text']}\n")

if __name__ == "__main__":
    main()
