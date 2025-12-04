"""Library-based classifier."""

from typing import Dict, List, Optional

from .base import BaseClassifier, ClassificationScore


class LibraryClassifier(BaseClassifier):
    """Classify based on metadata.library field."""

    LIBRARY_MAPPING = {
        "python": {
            "pandas", "numpy", "scipy", "matplotlib", "scikit-learn", "sklearn",
            "seaborn", "plotly", "sympy", "networkx", "pillow", "opencv", "cv2",
            "tensorflow", "torch", "keras", "jax", "statsmodels", "requests",
            "beautifulsoup", "bs4", "scrapy", "selenium", "flask", "django",
            "fastapi", "sqlalchemy", "pytest", "unittest", "asyncio", "aiohttp",
        },
        "sql": {"sql", "mysql", "postgresql", "sqlite", "oracle", "tsql", "plsql"},
        "javascript": {"javascript", "js", "node", "npm", "typescript", "ts", "react", "vue", "angular"},
        "java": {"java", "spring", "maven", "gradle", "junit"},
        "cpp": {"c++", "cpp", "boost", "cmake"},
        "csharp": {"c#", "csharp", ".net", "dotnet"},
        "go": {"go", "golang"},
        "rust": {"rust", "cargo"},
        "shell": {"bash", "shell", "sh", "zsh"},
        "docs": {"docs", "documentation", "markdown", "rst"},
    }

    def classify(self, item: Dict, available_domains: List[str]) -> Optional[ClassificationScore]:
        meta = item.get("metadata") or {}
        if not isinstance(meta, dict):
            return None

        lib = self.norm(meta.get("library"))
        if not lib:
            return None

        for domain, libs in self.LIBRARY_MAPPING.items():
            if lib in libs and domain in available_domains:
                return ClassificationScore(
                    domain=domain,
                    score=1.0,
                    reason=f"Library: {lib}"
                )

        return None

