"""Base classifier and score classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ClassificationScore:
    """Score for a domain classification."""
    domain: str
    score: float
    reason: str


class BaseClassifier(ABC):
    """Base class for domain classifiers."""

    @abstractmethod
    def classify(self, item: Dict, available_domains: List[str]) -> Optional[ClassificationScore]:
        """Classify an item to a domain."""
        pass

    def norm(self, s: Optional[str]) -> str:
        """Normalize string for comparison."""
        return (s or "").strip().lower()

