"""Complexity-based classifier."""

from typing import Dict, List, Optional

from .base import BaseClassifier, ClassificationScore


class ComplexityClassifier(BaseClassifier):
    """Classify based on code complexity and size."""

    def classify(self, item: Dict, available_domains: List[str]) -> Optional[ClassificationScore]:
        code = item.get("code", "")
        if not code:
            return None

        lines = len(code.split("\n"))
        complexity = len(code)

        if "python" in available_domains and complexity > 500 and lines > 20:
            return ClassificationScore(
                domain="python",
                score=0.3,
                reason="Complex code"
            )

        return None

