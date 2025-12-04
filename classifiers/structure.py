"""Code structure-based classifier."""

import re
from typing import Dict, List, Optional

from .base import BaseClassifier, ClassificationScore


class CodeStructureClassifier(BaseClassifier):
    """Classify based on code structure and syntax."""

    def classify(self, item: Dict, available_domains: List[str]) -> Optional[ClassificationScore]:
        code = self.norm(item.get("code", ""))
        if not code:
            return None

        if "sql" in available_domains and self._is_sql(code):
            return ClassificationScore(domain="sql", score=0.9, reason="SQL structure")

        if "python" in available_domains and self._is_python(code):
            return ClassificationScore(domain="python", score=0.85, reason="Python structure")

        if "javascript" in available_domains and self._is_javascript(code):
            return ClassificationScore(domain="javascript", score=0.85, reason="JavaScript structure")

        return None

    def _is_sql(self, code: str) -> bool:
        sql_patterns = [
            r"^\s*SELECT\s+", r"^\s*INSERT\s+INTO", r"^\s*UPDATE\s+",
            r"^\s*DELETE\s+FROM", r"^\s*CREATE\s+TABLE",
        ]
        return any(re.search(pattern, code, re.IGNORECASE | re.MULTILINE) for pattern in sql_patterns)

    def _is_python(self, code: str) -> bool:
        python_patterns = [
            r"^\s*def\s+\w+\s*\(", r"^\s*class\s+\w+", r"^\s*import\s+\w+",
            r"^\s*from\s+\w+\s+import", r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]",
        ]
        return any(re.search(pattern, code, re.MULTILINE) for pattern in python_patterns)

    def _is_javascript(self, code: str) -> bool:
        js_patterns = [
            r"^\s*function\s+\w+\s*\(", r"^\s*const\s+\w+\s*=",
            r"^\s*let\s+\w+\s*=", r"^\s*class\s+\w+\s*\{", r"=>\s*\{?",
        ]
        return any(re.search(pattern, code, re.MULTILINE) for pattern in js_patterns)

