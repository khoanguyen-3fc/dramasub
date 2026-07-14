"""Blind-translate source terms with an Ollama model — no dictionary hint — so a
judge can tell which terms the model already handles (drop) vs gets wrong (keep).

Deps: requests. Part of the `extract-dictionary` skill.

Usage:
  blind_translate.py --terms cand.json --model gemma4:latest \\
      --source-name Korean --target-name Vietnamese --out blind.json
  (--host defaults to $OLLAMA_HOST, else http://localhost:11434)

Input  : JSON list of source terms  ["출시", "미팅", ...]
Output : JSON list of {source, blind}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import requests


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--terms", required=True, help="JSON list of source-language terms")
    ap.add_argument("--model", required=True, help="Ollama model tag, e.g. gemma4:latest")
    ap.add_argument("--source-name", required=True, help="source language name, e.g. Korean")
    ap.add_argument("--target-name", required=True, help="target language name, e.g. Vietnamese")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    terms = json.load(open(args.terms))
    host = args.host if args.host.startswith("http") else "http://" + args.host
    print(f"{len(terms)} terms | model {args.model} | host {host}", file=sys.stderr)

    blind: dict[str, str] = {}
    for start in range(0, len(terms), args.batch):
        chunk = terms[start:start + args.batch]
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(chunk))
        prompt = (
            f"Dịch/Translate each {args.source_name} word or phrase into the most natural "
            f"{args.target_name} a TV-drama subtitle would use. Answer one per line, EXACTLY "
            f"in this format: `<number>. <{args.source_name}> => <{args.target_name}>`. "
            f"No explanations, nothing else.\n\n" + numbered
        )
        resp = ""
        for _ in range(3):
            try:
                r = requests.post(f"{host}/api/generate", json={
                    "model": args.model, "prompt": prompt, "stream": False,
                    "options": {"temperature": args.temperature}}, timeout=180)
                resp = r.json().get("response", "")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  retry ({exc})", file=sys.stderr)
                time.sleep(3)
        got = {}
        for line in resp.splitlines():
            m = re.match(r"\s*(\d+)\.\s*(.+?)\s*=>\s*(.+?)\s*$", line)
            if m:
                got[int(m.group(1))] = m.group(3).strip()
        for i, t in enumerate(chunk):
            blind[t] = got.get(i + 1, "")
        print(f"  {min(start + args.batch, len(terms))}/{len(terms)}", file=sys.stderr)

    json.dump([{"source": t, "blind": blind.get(t, "")} for t in terms],
              open(args.out, "w"), ensure_ascii=False, indent=1)
    missing = sum(1 for t in terms if not blind.get(t))
    print(f"wrote {args.out} | missing: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
