"""Advanced multi-strategy classifier."""

import re
from typing import Dict, List, Optional

from .base import BaseClassifier, ClassificationScore


class AdvancedClassifier(BaseClassifier):
    """Advanced classifier combining multiple strategies."""

    # Domain-specific patterns with weights
    DOMAIN_PATTERNS = {
        "python": {
            "strong": [
                r"import\s+(?:pandas|numpy|sklearn|tensorflow|torch|keras)",
                r"from\s+(?:pandas|numpy|sklearn|tensorflow|torch|keras)\s+import",
                r"\.py\b",
                r"def\s+\w+\s*\([^)]*\):",
                r"class\s+\w+\s*(?:\([^)]*\))?:",
            ],
            "medium": [
                r"import\s+\w+",
                r"from\s+\w+\s+import",
                r"if\s+__name__\s*==\s*['\"]__main__['\"]",
                r"lambda\s+",
                r"@\w+",  # decorators
            ],
            "weak": [
                r"print\s*\(",
                r"\[\s*\w+\s+for\s+\w+\s+in\s+",  # list comprehension
                r"\.append\(|\.extend\(|\.pop\(",
            ],
        },
        "sql": {
            "strong": [
                r"SELECT\s+.*\s+FROM",
                r"INSERT\s+INTO\s+\w+",
                r"UPDATE\s+\w+\s+SET",
                r"DELETE\s+FROM\s+\w+",
                r"CREATE\s+TABLE",
                r"ALTER\s+TABLE",
            ],
            "medium": [
                r"WHERE\s+",
                r"JOIN\s+",
                r"GROUP\s+BY",
                r"ORDER\s+BY",
                r"HAVING\s+",
            ],
            "weak": [
                r"SELECT\s+",
                r"FROM\s+",
                r"AND\s+",
                r"OR\s+",
            ],
        },
        "javascript": {
            "strong": [
                r"\.js\b",
                r"\.ts\b",
                r"function\s+\w+\s*\([^)]*\)\s*\{",
                r"const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
                r"import\s+.*\s+from\s+['\"]",
            ],
            "medium": [
                r"const\s+\w+\s*=",
                r"let\s+\w+\s*=",
                r"var\s+\w+\s*=",
                r"async\s+function",
                r"await\s+",
            ],
            "weak": [
                r"console\.",
                r"document\.",
                r"window\.",
            ],
        },
        "java": {
            "strong": [
                r"public\s+class\s+\w+",
                r"public\s+static\s+void\s+main",
                r"import\s+java\.",
                r"\.java\b",
            ],
            "medium": [
                r"public\s+\w+\s+\w+\s*\(",
                r"private\s+\w+\s+\w+",
                r"new\s+\w+\s*\(",
            ],
            "weak": [
                r"System\.out\.println",
                r"String\s+\w+\s*=",
            ],
        },
        "cpp": {
            "strong": [
                r"#include\s+<",
                r"\.cpp\b",
                r"\.h\b",
                r"std::\w+",
                r"namespace\s+\w+",
            ],
            "medium": [
                r"template\s*<",
                r"class\s+\w+\s*\{",
                r"public:",
                r"private:",
            ],
            "weak": [
                r"void\s+\w+\s*\(",
                r"int\s+\w+\s*=",
            ],
        },
        "shell": {
            "strong": [
                r"#!/bin/bash",
                r"#!/bin/sh",
                r"\.sh\b",
                r"for\s+\w+\s+in\s+",
                r"if\s+\[\s+",
            ],
            "medium": [
                r"\$\{?\w+\}?",
                r"echo\s+",
                r"grep\s+",
                r"sed\s+",
            ],
            "weak": [
                r"mkdir\s+",
                r"chmod\s+",
                r"ls\s+",
            ],
        },
    }

    def classify(self, item: Dict, available_domains: List[str]) -> Optional[ClassificationScore]:
        prompt = item.get("prompt", "")
        code = item.get("code", "")
        text = f"{prompt}\n{code}"

        best_score = None
        best_domain = None

        for domain in available_domains:
            if domain not in self.DOMAIN_PATTERNS:
                continue

            patterns = self.DOMAIN_PATTERNS[domain]
            score = 0.0

            # Strong patterns (weight: 3)
            for pattern in patterns.get("strong", []):
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    score += 3.0

            # Medium patterns (weight: 1.5)
            for pattern in patterns.get("medium", []):
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    score += 1.5

            # Weak patterns (weight: 0.5)
            for pattern in patterns.get("weak", []):
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    score += 0.5

            if score > 0:
                normalized_score = min(score / 15.0, 1.0)  # Normalize to 0-1
                if best_score is None or normalized_score > best_score:
                    best_score = normalized_score
                    best_domain = domain

        if best_score is not None and best_score > 0.3:  # Confidence threshold
            return ClassificationScore(
                domain=best_domain,
                score=best_score,
                reason=f"Advanced patterns (confidence: {best_score:.2f})"
            )

        return None

