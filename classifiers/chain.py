"""Classifier chain orchestrator."""

from typing import List, Optional, Tuple

from .base import BaseClassifier, ClassificationScore
from .complexity import ComplexityClassifier
from .keyword import KeywordClassifier
from .library import LibraryClassifier
from .structure import CodeStructureClassifier
from .advanced import AdvancedClassifier


class ClassifierChain:
    """Chain multiple classifiers with priority."""

    def __init__(self, classifiers: Optional[List[BaseClassifier]] = None):
        # Order matters: library is most reliable, then advanced patterns, then others
        self.classifiers = classifiers or [
            LibraryClassifier(),      # Highest priority: explicit library metadata
            AdvancedClassifier(),     # Strong pattern matching with weights
            CodeStructureClassifier(), # Code structure analysis
            KeywordClassifier(),      # Keyword-based classification
            ComplexityClassifier(),   # Fallback: complexity-based
        ]

    def classify(self, item: dict, available_domains: List[str], default_domain: str) -> Tuple[str, str]:
        """Classify item using chain of classifiers with confidence threshold."""
        best_score = None
        best_classifier_name = None

        for classifier in self.classifiers:
            try:
                score = classifier.classify(item, available_domains)
                if score and (best_score is None or score.score > best_score.score):
                    best_score = score
                    best_classifier_name = classifier.__class__.__name__
                    
                    # High confidence threshold - stop early if we find a strong match
                    if score.score >= 0.9:
                        break
            except Exception:
                continue

        if best_score and best_score.score >= 0.3:  # Minimum confidence threshold
            return best_score.domain, best_score.reason

        return default_domain, "Default fallback"

