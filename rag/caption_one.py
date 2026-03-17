# rag/caption_one.py
from __future__ import annotations
import argparse, base64, json, requests
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="llava:7b")
    ap.add_argument("--prompt", default="Describe this CT slice. Mention if lungs are visible and any obvious abnormalities. Do not diagnose.")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Missing image: {img_path}")

    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "images": [b64],
        "stream": False
    }

    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    print(data.get("response", "").strip())

if __name__ == "__main__":
    main()
