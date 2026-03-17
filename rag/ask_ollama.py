# rag/ask_ollama.py
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

def load_jsonl(path: Path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

def retrieve(query: str, k: int, pairs_path: Path, emb_path: Path, device: str):
    pairs = list(load_jsonl(pairs_path))
    E = np.load(emb_path)["embeddings"].astype(np.float32)
    if len(pairs) != E.shape[0]:
        raise SystemExit(f"pairs ({len(pairs)}) != embeddings ({E.shape[0]})")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32)[0]
    scores = E @ q

    topk = np.argpartition(-scores, k)[:k]
    topk = topk[np.argsort(-scores[topk])]
    return [(int(i), float(scores[i]), pairs[int(i)]) for i in topk]

def call_ollama(model: str, prompt: str) -> str:
    p = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "ollama failed")
    return p.stdout.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pairs", default="data/metadata/pairs.jsonl")
    ap.add_argument("--text_emb", default="data/embeddings/text_embeddings.npz")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--llm", default="llama3:latest")
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    emb_path = Path(args.text_emb)
    if not pairs_path.exists():
        raise SystemExit(f"Missing pairs: {pairs_path}")
    if not emb_path.exists():
        raise SystemExit(f"Missing text embeddings: {emb_path}")

    hits = retrieve(args.question, args.k, pairs_path, emb_path, args.device)

    context = []
    for rank, (_, score, r) in enumerate(hits, 1):
        context.append(
            f"[{rank}] score={score:.4f} doc_id={r['doc_id']}\n"
            f"meta={r['text']}\n"
            f"image_path={r['image_path']}\n"
            f"dicom_path={r['dicom_path']}\n"
        )

    prompt = (
        "You are a medical imaging assistant.\n"
        "Use ONLY the provided metadata context. Do not diagnose.\n"
        "If context is insufficient, say so.\n\n"
        "CONTEXT:\n" + "\n".join(context) + "\n"
        "QUESTION:\n" + args.question + "\n\n"
        "ANSWER:\n"
    )

    print(call_ollama(args.llm, prompt))

if __name__ == "__main__":
    main()
