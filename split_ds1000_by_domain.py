#!/usr/bin/env python3
"""
Split DS1000 (JSONL) into domain-specific JSONL files for code-generation experiments.

Default domains: python, sql, docs

Heuristics:
- Prefer metadata.library if present. Map common DS1000 libraries to domains.
- Otherwise, route by prompt keywords.
- If still ambiguous, send to 'misc' (optional) or default fallback domain ('python').

Usage:
  python split_ds1000_by_domain.py \
      --input dataset/test.jsonl \
      --outdir dataset/domains \
      --domains python,sql,docs \
      --create-misc

Outputs (example):
  dataset/domains/python.jsonl
  dataset/domains/sql.jsonl
  dataset/domains/docs.jsonl
  dataset/domains/misc.jsonl   (if --create-misc)
  dataset/domains/summary.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def build_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split DS1000 JSONL into domains")
    p.add_argument("--input", type=str, default="dataset/ds1000.jsonl", help="Input DS1000 JSONL file")
    p.add_argument("--outdir", type=str, default="dataset/domains", help="Output directory for domain JSONL files")
    p.add_argument(
        "--domains",
        type=str,
        default="python,sql,docs",
        help="Comma-separated domain list to create (default: python,sql,docs)",
    )
    p.add_argument(
        "--create-misc",
        action="store_true",
        help="If set, create misc.jsonl and route unassigned items there (default: fallback to python)",
    )
    return p.parse_args()


def norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def choose_domain(item: Dict, domains: List[str], create_misc: bool) -> str:
    # 1) metadata.library mapping if present
    lib = None
    meta = item.get("metadata") or {}
    if isinstance(meta, dict):
        lib = norm(meta.get("library"))

    # Map common DS1000 libraries to our domains
    if lib:
        if lib in {"pandas", "numpy", "numPy", "python", "matplotlib", "scipy", "sklearn", "scikit-learn"}:
            if "python" in domains:
                return "python"
        if lib in {"sql"}:
            if "sql" in domains:
                return "sql"
        if lib in {"docs", "documentation"}:
            if "docs" in domains:
                return "docs"

    # 2) keyword routing from prompt
    prompt = norm(item.get("prompt"))

    kw = {
        "python": [
            "python", "pandas", "numpy", "dataframe", "series", "index", "matplotlib",
            "np.", "pd.", "dict", "list", "tuple", "itertools", "comprehension",
        ],
        "sql": [
            "sql", "select", "update", "delete", "insert", "join", "group by", "where",
            "table", "database", "query", "from ", "inner join", "left join",
        ],
        "docs": [
            "documentation", "docstring", "readme", "comment", "write docs", "explain",
        ],
    }

    for d in domains:
        for token in kw.get(d, []):
            if token in prompt:
                return d

    # 3) fallback
    if create_misc:
        return "misc"
    # Otherwise default to python if available, else first domain
    return "python" if "python" in domains else domains[0]


def main() -> int:
    args = build_argparser()
    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    domains: List[str] = [d.strip() for d in args.domains.split(",") if d.strip()]
    if args.create_misc and "misc" not in domains:
        domains_with_misc = domains + ["misc"]
    else:
        domains_with_misc = domains

    # Open writers per domain
    writers: Dict[str, any] = {}
    for d in domains_with_misc:
        writers[d] = open(outdir / f"{d}.jsonl", "w", encoding="utf-8")

    counts: Dict[str, int] = {d: 0 for d in domains_with_misc}
    total = 0

    if not inp.exists():
        print(f"[ERROR] Input not found: {inp}")
        return 1

    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            dom = choose_domain(item, domains, args.create_misc)
            writers[dom].write(json.dumps(item, ensure_ascii=False) + "\n")
            counts[dom] = counts.get(dom, 0) + 1

    # Close writers
    for w in writers.values():
        w.close()

    # Summary
    summary_lines = [
        f"Input: {inp}",
        f"Total: {total}",
        f"Domains: {', '.join(domains_with_misc)}",
        "Counts:",
    ]
    for d in domains_with_misc:
        summary_lines.append(f"  - {d}: {counts.get(d, 0)}")

    (outdir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

