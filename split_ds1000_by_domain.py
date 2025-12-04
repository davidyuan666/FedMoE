#!/usr/bin/env python3
"""Split DS1000 JSONL into domain-specific files."""

from __future__ import annotations

import argparse
import json
import re
import sys
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from classifiers import ClassifierChain
except Exception as e:
    # Fallback enhanced classifier focused on code content (libraries/APIs/tasks)
    class ClassifierChain:  # type: ignore
        def __init__(self) -> None:
            # Fields
            self.code_like_keys = (
                "reference_code",
                "code",
                "code_context",
                "solution",
                "program",
                "generated_code",
            )
            self.text_keys = (
                "prompt",
                "question",
                "instruction",
                "text",
                "problem",
                "task",
                "description",
                "title",
            )
            # Library normalization
            self.lib_syn = {
                # dataframes
                "pandas": "pandas",
                "pd": "pandas",
                # arrays
                "numpy": "numpy",
                "np": "numpy",
                # viz
                "matplotlib": "matplotlib",
                "pyplot": "matplotlib",
                "seaborn": "matplotlib",
                "sns": "matplotlib",
                # ml/science
                "scikit-learn": "sklearn",
                "sklearn": "sklearn",
                "scipy": "scipy",
                "statsmodels": "statsmodels",
                # dl
                "pytorch": "torch",
                "torch": "torch",
                "tensorflow": "tensorflow",
                "tf": "tensorflow",
                "keras": "tensorflow",
                # big data
                "pyspark": "pyspark",
                "spark": "pyspark",
                # gradient boosting
                "xgboost": "xgboost",
                "lightgbm": "lightgbm",
                "catboost": "catboost",
                # sql
                "sql": "sql",
                "sqlite": "sql",
                "postgresql": "sql",
                "mysql": "sql",
                # docs
                "markdown": "docs",
                "readme": "docs",
            }
            # Regex for imports/usages
            self.patterns = {
                "pandas": re.compile(r"\b(import|from)\s+pandas\b|\bpd\.|\bDataFrame\b|\bSeries\b|\b(read_csv|read_parquet|read_json)\s*\(|\b(groupby|merge|join|concat|pivot_table|melt|stack|unstack|assign|value_counts|astype|sort_values|dropna|fillna|loc|iloc)\b", re.I),
                "numpy": re.compile(r"\b(import|from)\s+numpy\b|\bnp\.", re.I),
                "matplotlib": re.compile(r"\b(import|from)\s+matplotlib\b|\bplt\.|\bseaborn\b|\bsns\.", re.I),
                "sklearn": re.compile(r"\b(import|from)\s+sklearn\b|\bscikit-learn\b", re.I),
                "scipy": re.compile(r"\b(import|from)\s+scipy\b|\bscipy\.", re.I),
                "statsmodels": re.compile(r"\b(import|from)\s+statsmodels\b|\bsm\.", re.I),
                "torch": re.compile(r"\b(import|from)\s+torch\b|\btorch\.", re.I),
                "tensorflow": re.compile(r"\b(import|from)\s+tensorflow\b|\btf\.", re.I),
                "pyspark": re.compile(r"\b(import|from)\s+pyspark\b|\bSparkSession\b", re.I),
                "xgboost": re.compile(r"\b(import|from)\s+xgboost\b|\bxgb\.", re.I),
                "lightgbm": re.compile(r"\b(import|from)\s+lightgbm\b|\blgb\.", re.I),
                "catboost": re.compile(r"\b(import|from)\s+catboost\b", re.I),
                "sql": re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN|GROUP\s+BY|ORDER\s+BY)\b", re.I),
                "docs": re.compile(r"^\s*#\s+\w+|\bmarkdown\b|\breadme\b", re.M|re.I),
            }

        def _gather(self, item) -> dict:
            fields = {}
            if not isinstance(item, dict):
                return fields
            for k in list(self.code_like_keys) + list(self.text_keys):
                v = item.get(k)
                if isinstance(v, str) and v:
                    fields[k] = v
            # metadata.library hint
            meta = item.get("metadata") or {}
            if isinstance(meta, dict):
                lib = meta.get("library")
                if isinstance(lib, str) and lib:
                    fields["metadata.library"] = lib
            return fields

        def _norm_lib(self, name: str) -> str:
            name = name.strip().lower()
            return self.lib_syn.get(name, name)

        def _pick_allowed(self, candidates: List[str], allowed: set[str]) -> Optional[str]:
            for c in candidates:
                if c in allowed:
                    return c
            return None

        def classify(self, item, domains, default_domain):
            allowed: set[str] = set(domains)
            fields = self._gather(item)

            # 1) metadata.library first
            meta_lib = fields.get("metadata.library")
            if isinstance(meta_lib, str):
                norm = self._norm_lib(meta_lib)
                # common library -> broader buckets if needed
                mapping = {
                    "pandas": ["pandas", "python"],
                    "numpy": ["numpy", "python"],
                    "matplotlib": ["matplotlib", "viz", "python"],
                    "sklearn": ["sklearn", "ml", "python"],
                    "scipy": ["scipy", "python"],
                    "statsmodels": ["statsmodels", "python"],
                    "torch": ["torch", "pytorch", "dl", "python"],
                    "tensorflow": ["tensorflow", "keras", "dl", "python"],
                    "pyspark": ["pyspark", "spark", "bigdata"],
                    "xgboost": ["xgboost", "gbdt", "ml", "python"],
                    "lightgbm": ["lightgbm", "gbdt", "ml", "python"],
                    "catboost": ["catboost", "gbdt", "ml", "python"],
                    "sql": ["sql"],
                    "docs": ["docs"],
                }
                for key, buckets in mapping.items():
                    if norm == key:
                        chosen = self._pick_allowed(buckets, allowed)
                        if chosen:
                            return chosen, f"metadata.library:{norm}->{chosen}"
                        break

            # Concatenate code and text
            code_concat = "\n".join(fields.get(k, "") for k in self.code_like_keys if k in fields)
            text_concat = "\n".join(fields.get(k, "") for k in self.text_keys if k in fields)
            all_concat = (code_concat + "\n" + text_concat)

            # 2) Library import/usage patterns
            scores: Dict[str, int] = defaultdict(int)
            for lib, pat in self.patterns.items():
                if pat.search(all_concat):
                    # Score the direct lib bucket and its broader aliases
                    lib_map = {
                        "pandas": ["pandas", "python"],
                        "numpy": ["numpy", "python"],
                        "matplotlib": ["matplotlib", "viz", "python"],
                        "sklearn": ["sklearn", "ml", "python"],
                        "scipy": ["scipy", "python"],
                        "statsmodels": ["statsmodels", "python"],
                        "torch": ["torch", "pytorch", "dl", "python"],
                        "tensorflow": ["tensorflow", "keras", "dl", "python"],
                        "pyspark": ["pyspark", "spark", "bigdata"],
                        "xgboost": ["xgboost", "gbdt", "ml", "python"],
                        "lightgbm": ["lightgbm", "gbdt", "ml", "python"],
                        "catboost": ["catboost", "gbdt", "ml", "python"],
                        "sql": ["sql"],
                        "docs": ["docs"],
                    }
                    for b in lib_map.get(lib, [lib]):
                        scores[b] += 1
            if scores:
                # pick highest-scoring allowed bucket
                ordered = sorted(scores.items(), key=lambda x: -x[1])
                for b, _ in ordered:
                    if b in allowed:
                        return b, f"pattern:{b}"

            # 3) Fallback
            fallback = default_domain if default_domain in domains else domains[0]
            return fallback, "fallback:default"


def build_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split DS1000 JSONL into domains")
    p.add_argument("--input", type=str, default="dataset/ds1000.jsonl", help="Input JSONL file")
    p.add_argument("--outdir", type=str, default="dataset/domains", help="Output directory")
    p.add_argument(
        "--domains",
        type=str,
        default="pandas,numpy,sklearn,sql,py_core",
        help="Comma-separated domains (default: 5 content categories)",
    )
    p.add_argument("--create-misc", action="store_true", help="Create misc.jsonl")
    p.add_argument("--default-domain", type=str, default="py_core", help="Default fallback domain")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    p.add_argument("--sample-size", type=int, default=None, help="Process only first N items")
    return p.parse_args()


class SimpleStats:
    """Lightweight statistics collector."""
    
    def __init__(self):
        self.counts: Dict[str, int] = defaultdict(int)
        self.reasons: Dict[str, int] = defaultdict(int)
        self.total = 0
    
    def record(self, domain: str, reason: str) -> None:
        self.counts[domain] += 1
        self.reasons[reason] += 1
        self.total += 1
    
    def get_summary(self) -> Dict:
        return {
            "total": self.total,
            "counts": dict(self.counts),
            "top_reasons": dict(sorted(self.reasons.items(), key=lambda x: x[1], reverse=True)[:10]),
        }


class DS1000Splitter:
    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        domains: List[str],
        create_misc: bool = False,
        default_domain: str = "python",
        verbose: bool = False,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.domains = domains
        self.create_misc = create_misc
        self.default_domain = default_domain
        self.verbose = verbose
        self.domains_with_misc = (
            domains + ["misc"] if create_misc and "misc" not in domains else domains
        )
        self.classifier_chain = ClassifierChain()
        self.stats = SimpleStats()
        self.writers: Dict[str, any] = {}

    def validate(self) -> bool:
        if not self.input_path.exists():
            print(f"[ERROR] Input not found: {self.input_path}")
            return False
        if self.default_domain not in self.domains_with_misc:
            print(f"[ERROR] Default domain '{self.default_domain}' not in domains")
            return False
        return True

    def setup_writers(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for domain in self.domains_with_misc:
            output_file = self.output_dir / f"{domain}.jsonl"
            self.writers[domain] = open(output_file, "w", encoding="utf-8")
            if self.verbose:
                print(f"[SETUP] Created writer for: {domain}")

    def close_writers(self) -> None:
        for writer in self.writers.values():
            writer.close()

    def process(self, sample_size: Optional[int] = None) -> int:
        if not self.validate():
            return 1

        self.setup_writers()

        processed = 0
        try:
            with open(self.input_path, "r", encoding="utf-8") as f:
                # Detect JSON array vs JSONL by peeking the first non-whitespace char
                pos = f.tell()
                head = f.read(4096)
                f.seek(pos)
                first_non_ws = next((ch for ch in head if not ch.isspace()), "")

                if first_non_ws == "[":
                    # JSON array input
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Failed to parse JSON array: {e}")
                        return 1

                    if not isinstance(data, list):
                        print(f"[ERROR] JSON input is not an array. Got: {type(data).__name__}")
                        return 1

                    if sample_size is not None:
                        data = data[:sample_size]

                    if self.verbose:
                        print(f"[MODE] Detected JSON array with {len(data)} items")

                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        domain, reason = self.classifier_chain.classify(
                            item, self.domains_with_misc, self.default_domain
                        )
                        self.writers[domain].write(json.dumps(item, ensure_ascii=False) + "\n")
                        self.stats.record(domain, reason)
                        processed += 1
                        if self.verbose and processed % 100 == 0:
                            print(f"[PROGRESS] Processed {processed} items...")
                else:
                    # JSONL input (one JSON object per line)
                    if self.verbose:
                        print("[MODE] Detected JSONL (one object per line)")
                    for idx, line in enumerate(f):
                        if sample_size is not None and idx >= sample_size:
                            break

                        line = line.strip()
                        if not line:
                            continue

                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            # Skip malformed lines silently to mimic original behavior
                            continue

                        if not isinstance(item, dict):
                            continue

                        domain, reason = self.classifier_chain.classify(
                            item, self.domains_with_misc, self.default_domain
                        )

                        self.writers[domain].write(json.dumps(item, ensure_ascii=False) + "\n")
                        self.stats.record(domain, reason)
                        processed += 1

                        if self.verbose and processed % 100 == 0:
                            print(f"[PROGRESS] Processed {processed} items...")

        finally:
            self.close_writers()

        if processed == 0:
            print("[WARN] No items processed. Make sure the input is JSONL (one JSON object per line) or a JSON array file. Use --verbose for details.")

        return 0

    def generate_reports(self) -> None:
        summary = self.stats.get_summary()
        total = summary["total"]
        
        # Print summary
        print("\n" + "=" * 80)
        print("SPLITTING SUMMARY")
        print("=" * 80)
        print(f"Total items: {total}")
        print(f"Domains: {', '.join(self.domains_with_misc)}\n")
        
        print("Distribution:")
        for domain in self.domains_with_misc:
            count = summary["counts"].get(domain, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {domain:15} {count:6} items ({pct:5.1f}%)")
        
        self._print_balance_analysis(summary)

    def _print_balance_analysis(self, summary: Dict) -> None:
        counts = list(summary["counts"].values())
        if not counts:
            return
        
        total = summary["total"]
        avg = total / len(counts)
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        is_balanced = imbalance_ratio < 2.0
        
        print("\n" + "=" * 80)
        print("BALANCE ANALYSIS")
        print("=" * 80)
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")
        print(f"Is balanced: {'Yes' if is_balanced else 'No'}")
        
        # Suggest improvements
        suggestions = []
        for domain, count in summary["counts"].items():
            pct = (count / total * 100) if total > 0 else 0
            if count < avg * 0.5:
                suggestions.append(
                    f"Domain '{domain}' is underrepresented ({count} items, {pct:.1f}%)"
                )
            elif count > avg * 1.5:
                suggestions.append(
                    f"Domain '{domain}' is overrepresented ({count} items, {pct:.1f}%)"
                )
        
        if suggestions:
            print("\nSuggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")


def main() -> int:
    args = build_argparser()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not domains:
        print("[ERROR] No domains specified")
        return 1

    splitter = DS1000Splitter(
        input_path=Path(args.input),
        output_dir=Path(args.outdir),
        domains=domains,
        create_misc=args.create_misc,
        default_domain=args.default_domain,
        verbose=args.verbose,
    )

    print(f"[INFO] Starting DS1000 splitting...")
    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Output: {args.outdir}")
    print(f"[INFO] Domains: {', '.join(domains)}")
    print()

    result = splitter.process(sample_size=args.sample_size)
    if result != 0:
        return result

    splitter.generate_reports()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
