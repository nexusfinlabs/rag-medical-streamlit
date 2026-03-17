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
    ap.add_argument("--pairs", default="data/metadata/pairs_labeled.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise SystemExit(f"Missing {pairs_path} (run pseudo_label.py first)")

    rows, texts = [], []
    for line in pairs_path.open():
        r = json.loads(line)
        rows.append(r)
        # IMPORTANT: metemos también labels como “texto” para que el query las encuentre
        labels = " ".join(r.get("labels", []))
        texts.append(f"{labels} | {r.get('text','')}")

    model = SentenceTransformer(args.model, device=args.device)
    X = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    q = model.encode([args.query], normalize_embeddings=True)[0]

    scores = X @ q
    k = min(args.k, len(scores))
    idx = np.argsort(-scores)[:k]

    print(f"\nQuery: {args.query}\nTop-{k} results:\n")
    for rank, i in enumerate(idx, 1):
        r = rows[int(i)]
        print(f"{rank:02d}) score={float(scores[i]):.4f}")
        print(f"    image: {r['image_path']}")
        print(f"    dicom:  {r['dicom_path']}")
        print(f"    labels: {r.get('labels',[])}")
        print(f"    text:   {r['text']}\n")

if __name__ == "__main__":
    main()
