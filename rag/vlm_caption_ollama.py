#!/usr/bin/env python3
import base64, argparse, sys
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()

def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Describe this CT chest slice. Do not diagnose.")
    ap.add_argument("--model", default="llava:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=1800)   # 30 min
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Missing image: {img_path}")

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "images": [b64_image(img_path)],
        "stream": bool(args.stream),
    }

    if args.stream:
        with requests.post(f"{args.host}/api/generate", json=payload, stream=True, timeout=(10, args.timeout)) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                j = __import__("json").loads(line)
                chunk = j.get("response", "")
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            print()
    else:
        r = requests.post(f"{args.host}/api/generate", json=payload, timeout=(10, args.timeout))
        r.raise_for_status()
        print(r.json().get("response","").strip())

if __name__ == "__main__":
    main()
