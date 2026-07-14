"""Write a dramasub dictionary YAML from kept entries, with a YAML-safe scalar
serializer (double-quotes only when a value would otherwise mis-parse).

Deps: PyYAML (validation only). Part of the `extract-dictionary` skill.

Usage:
  write_dictionary.py --entries kept.json --pair ko-vi --version 1 \\
      --header "one-line provenance (NO work/show names)" --out dictionary.ko-vi.yaml

Input : JSON list of {source, target, note?}
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml


def scalar(s: str) -> str:
    """Plain scalar when safe, else a JSON (double-quoted) scalar — valid YAML."""
    if s == "" or s[0] in "\"'>|@`%&*!?#,[]{}-" or any(c in s for c in ":#") or s != s.strip():
        return json.dumps(s, ensure_ascii=False)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entries", required=True, help="JSON list of {source,target,note?}")
    ap.add_argument("--pair", required=True, help="language pair, e.g. ko-vi")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--header", default="", help="provenance note; '\\n' splits lines; NO work names")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    entries = json.load(open(args.entries))
    lines = [f"# dramasub dictionary — {args.pair}", f"# version: {args.version}"]
    for ln in args.header.split("\\n") if args.header else []:
        lines.append(f"#   {ln}")
    lines += [f"version: {args.version}", "terms:"]
    for e in entries:
        lines.append(f"  - source: {scalar(e['source'])}")
        lines.append(f"    target: {scalar(e['target'])}")
        if e.get("note"):
            lines.append(f"    note: {scalar(e['note'])}")
    open(args.out, "w").write("\n".join(lines) + "\n")

    d = yaml.safe_load(open(args.out))
    srcs = [t["source"] for t in d["terms"]]
    dups = sorted(s for s in set(srcs) if srcs.count(s) > 1)
    print(f"wrote {args.out}: {len(d['terms'])} terms, version {d['version']}"
          + (f" | DUPLICATES: {dups}" if dups else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
