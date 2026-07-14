"""Align parallel subtitle pairs (source language A <-> reference language B) by
timestamp overlap, producing per-work (a, b) term-mining corpora.

Deps: pysubs2. Part of the `extract-dictionary` skill.

Usage:
  align_pairs.py --dir SUBS --a-suffix .kor.srt --b-suffix .vie.srt --out aligned.json
  align_pairs.py --pairs a1.srt:b1.srt,a2.srt:b2.srt --out aligned.json
  (optionally) --known dictionary.yaml   # embeds existing sources so miners skip them
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pysubs2


def plain(ev) -> str:
    return ev.plaintext.replace("\n", " ").strip()


def align(a_path: str, b_path: str) -> list[dict]:
    a, b = pysubs2.load(a_path), pysubs2.load(b_path)
    out = []
    for s in a.events:
        if not plain(s):
            continue
        vi = " ".join(plain(r) for r in b.events if r.start < s.end and r.end > s.start)
        if vi:
            out.append({"a": plain(s), "b": vi})
    return out


def known_sources(path: str) -> list[str]:
    import yaml
    data = yaml.safe_load(open(path))
    terms = data.get("terms", data) if isinstance(data, dict) else data
    return sorted({t["source"] for t in terms if t.get("source")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", help="directory holding the A and B subtitle files")
    ap.add_argument("--a-suffix", default=".kor.srt", help="source-file suffix")
    ap.add_argument("--b-suffix", default=".vie.srt", help="reference-file suffix")
    ap.add_argument("--pairs", help="explicit 'A:B,A:B' list (overrides --dir)")
    ap.add_argument("--known", help="existing dictionary YAML; its sources are recorded")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    jobs = []  # (name, a_path, b_path)
    if args.pairs:
        for p in args.pairs.split(","):
            a, b = p.split(":")
            jobs.append((os.path.basename(a).replace(args.a_suffix, ""), a, b))
    elif args.dir:
        for a in sorted(glob.glob(os.path.join(args.dir, f"*{args.a_suffix}"))):
            b = a[: -len(args.a_suffix)] + args.b_suffix
            if os.path.exists(b):
                jobs.append((os.path.basename(a)[: -len(args.a_suffix)], a, b))
    else:
        sys.exit("give --dir or --pairs")
    if not jobs:
        sys.exit("no A/B pairs found")

    files = {}
    for name, a, b in jobs:
        files[name] = align(a, b)
        print(f"  {name}: {len(files[name])} aligned pairs", file=sys.stderr)

    payload = {"known_sources": known_sources(args.known) if args.known else [], "files": files}
    json.dump(payload, open(args.out, "w"), ensure_ascii=False, indent=1)
    total = sum(len(v) for v in files.values())
    print(f"wrote {args.out}: {total} pairs across {len(files)} works", file=sys.stderr)


if __name__ == "__main__":
    main()
