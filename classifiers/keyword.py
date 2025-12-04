"""Keyword-based classifier."""

import re
from typing import Dict, List, Optional

from .base import BaseClassifier, ClassificationScore


class KeywordClassifier(BaseClassifier):
    """Classify based on keywords in prompt and code."""

    KEYWORDS = {
        "python": {
            "keywords": [
                "python", "pandas", "numpy", "dataframe", "series", "matplotlib",
                "np.", "pd.", "dict", "list", "tuple", "itertools", "lambda",
                "decorator", "class", "def ", "import ", "module", "sklearn",
            ],
            "patterns": [r"\.py\b", r"import\s+\w+", r"def\s+\w+\s*\(", r"class\s+\w+"],
        },
        "sql": {
            "keywords": [
                "sql", "select", "update", "delete", "insert", "join", "group by",
                "where", "table", "database", "query", "from ", "order by",
            ],
            "patterns": [r"SELECT\s+", r"INSERT\s+INTO", r"UPDATE\s+", r"DELETE\s+FROM"],
        },
        "javascript": {
            "keywords": [
                "javascript", "js", "node", "react", "vue", "angular", "typescript",
                "const ", "let ", "var ", "function ", "=>", "async", "await",
            ],
            "patterns": [r"\.js\b", r"\.ts\b", r"function\s+\w+\s*\(", r"const\s+\w+\s*="],
        },
        "java": {
            "keywords": [
                "java", "class ", "public ", "private ", "static ", "void ",
                "string", "integer", "arraylist", "spring", "maven",
            ],
            "patterns": [r"\.java\b", r"public\s+class\s+\w+", r"import\s+java\."],
        },
        "cpp": {
            "keywords": [
                "c++", "cpp", "std::", "vector", "string", "pointer", "template",
                "class ", "namespace", "iostream",
            ],
            "patterns": [r"\.cpp\b", r"\.h\b", r"#include", r"std::"],
        },
        "shell": {
            "keywords": [
                "bash", "shell", "sh", "command", "script", "echo", "grep",
                "sed", "awk", "pipe", "chmod", "mkdir",
            ],
            "patterns": [r"\.sh\b", r"#!/bin/bash", r"\$\{?\w+\}?"],
        },
        "docs": {
            "keywords": [
                "documentation", "docstring", "readme", "comment", "explain",
                "description", "tutorial", "guide", "markdown", "rst",
            ],
            "patterns": [r"\.md\b", r"\.rst\b", r"\"\"\".*?\"\"\""],
        },
    }

    def classify(self, item: Dict, available_domains: List[str]) -> Optional[ClassificationScore]:
        prompt = self.norm(item.get("prompt", ""))
        code = self.norm(item.get("code", ""))
        text = f"{prompt} {code}"

        best_score = None

        for domain in available_domains:
            if domain not in self.KEYWORDS:
                continue

            kw_config = self.KEYWORDS[domain]
            score = 0.0

            for keyword in kw_config.get("keywords", []):
                if keyword in text:
                    score += 1.0

            for pattern in kw_config.get("patterns", []):
                if re.search(pattern, text, re.IGNORECASE):
                    score += 2.0

            if score > 0:
                normalized_score = min(score / 10.0, 1.0)
                if best_score is None or normalized_score > best_score.score:
                    best_score = ClassificationScore(
                        domain=domain,
                        score=normalized_score,
                        reason=f"Keywords (score: {score:.1f})"
                    )

        return best_score

