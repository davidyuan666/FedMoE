"""
Statistics and analysis for DS1000 dataset splitting.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DomainStats:
    """Statistics for a single domain."""
    domain: str
    count: int = 0
    avg_prompt_len: float = 0.0
    avg_code_len: float = 0.0
    avg_lines: float = 0.0
    libraries: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    top_keywords: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "domain": self.domain,
            "count": self.count,
            "avg_prompt_len": round(self.avg_prompt_len, 2),
            "avg_code_len": round(self.avg_code_len, 2),
            "avg_lines": round(self.avg_lines, 2),
            "top_libraries": dict(sorted(self.libraries.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_keywords": dict(sorted(self.top_keywords.items(), key=lambda x: x[1], reverse=True)[:10]),
        }


class StatisticsCollector:
    """Collect statistics during dataset splitting."""

    def __init__(self):
        self.stats: Dict[str, DomainStats] = defaultdict(lambda: DomainStats(domain=""))
        self.total_items = 0
        self.classification_reasons: Dict[str, int] = defaultdict(int)

    def record_item(
        self,
        domain: str,
        item: Dict,
        reason: str = "",
    ) -> None:
        """Record an item classification."""
        if domain not in self.stats:
            self.stats[domain] = DomainStats(domain=domain)

        stats = self.stats[domain]
        stats.count += 1
        self.total_items += 1

        # Update lengths
        prompt = item.get("prompt", "")
        code = item.get("code", "")
        stats.avg_prompt_len = (
            (stats.avg_prompt_len * (stats.count - 1) + len(prompt)) / stats.count
        )
        stats.avg_code_len = (
            (stats.avg_code_len * (stats.count - 1) + len(code)) / stats.count
        )
        lines = len(code.split("\n")) if code else 0
        stats.avg_lines = (
            (stats.avg_lines * (stats.count - 1) + lines) / stats.count
        )

        # Track libraries
        meta = item.get("metadata") or {}
        if isinstance(meta, dict):
            lib = (meta.get("library") or "").strip().lower()
            if lib:
                stats.libraries[lib] += 1

        # Track classification reasons
        if reason:
            self.classification_reasons[reason] += 1

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "total_items": self.total_items,
            "domains": {
                domain: stats.to_dict()
                for domain, stats in sorted(self.stats.items())
            },
            "classification_reasons": dict(
                sorted(
                    self.classification_reasons.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
            ),
        }

    def save_report(self, output_path: Path) -> None:
        """Save detailed statistics report."""
        summary = self.get_summary()
        
        report_lines = [
            "=" * 80,
            "DS1000 Dataset Splitting Report",
            "=" * 80,
            "",
            f"Total Items: {summary['total_items']}",
            "",
            "Domain Distribution:",
            "-" * 80,
        ]

        # Calculate percentages
        total = summary["total_items"]
        for domain_name, domain_stats in summary["domains"].items():
            count = domain_stats["count"]
            pct = (count / total * 100) if total > 0 else 0
            report_lines.append(
                f"  {domain_name:15} {count:6} items ({pct:5.1f}%)"
            )
            report_lines.append(
                f"    - Avg prompt len: {domain_stats['avg_prompt_len']:.0f} chars"
            )
            report_lines.append(
                f"    - Avg code len:   {domain_stats['avg_code_len']:.0f} chars"
            )
            report_lines.append(
                f"    - Avg lines:      {domain_stats['avg_lines']:.1f} lines"
            )
            if domain_stats["top_libraries"]:
                libs = ", ".join(
                    f"{lib}({cnt})"
                    for lib, cnt in list(domain_stats["top_libraries"].items())[:5]
                )
                report_lines.append(f"    - Top libraries:  {libs}")
            report_lines.append("")

        report_lines.extend([
            "Top Classification Reasons:",
            "-" * 80,
        ])
        for reason, count in list(summary["classification_reasons"].items())[:15]:
            pct = (count / total * 100) if total > 0 else 0
            report_lines.append(f"  {reason:40} {count:6} ({pct:5.1f}%)")

        report_text = "\n".join(report_lines)
        output_path.write_text(report_text, encoding="utf-8")
        print(report_text)

    def save_json_report(self, output_path: Path) -> None:
        """Save JSON format report."""
        summary = self.get_summary()
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )


class BalanceAnalyzer:
    """Analyze and suggest domain balance improvements."""

    @staticmethod
    def analyze_balance(stats: Dict[str, DomainStats]) -> Dict:
        """Analyze domain balance."""
        if not stats:
            return {}

        counts = [s.count for s in stats.values()]
        total = sum(counts)
        avg = total / len(counts) if counts else 0

        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')

        return {
            "total_items": total,
            "num_domains": len(counts),
            "avg_per_domain": round(avg, 2),
            "min_count": min(counts),
            "max_count": max(counts),
            "imbalance_ratio": round(imbalance_ratio, 2),
            "is_balanced": imbalance_ratio < 2.0,  # Consider balanced if ratio < 2
        }

    @staticmethod
    def suggest_improvements(stats: Dict[str, DomainStats]) -> List[str]:
        """Suggest improvements for better balance."""
        suggestions = []
        
        if not stats:
            return suggestions

        counts = {domain: s.count for domain, s in stats.items()}
        total = sum(counts.values())
        avg = total / len(counts)

        for domain, count in counts.items():
            pct = (count / total * 100) if total > 0 else 0
            if count < avg * 0.5:
                suggestions.append(
                    f"Domain '{domain}' is underrepresented ({count} items, {pct:.1f}%). "
                    f"Consider adjusting keywords or adding more specific patterns."
                )
            elif count > avg * 1.5:
                suggestions.append(
                    f"Domain '{domain}' is overrepresented ({count} items, {pct:.1f}%). "
                    f"Consider moving some items to other domains."
                )

        return suggestions

