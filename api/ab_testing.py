"""
A/B Testing logic for model version selection
"""

import random
import logging
from typing import Dict
from models.sentiment import SentimentModel

logger = logging.getLogger(__name__)


class ABTestRouter:
    """
    Router for A/B testing different model versions.
    Implements random 50/50 traffic split between versions.
    """

    def __init__(self, versions: list = None):
        """
        Initialize A/B test router.

        Args:
            versions: List of model versions to test (default: ["v1", "v2"])
        """
        self.versions = versions or ["v1", "v2"]
        self._selection_counts = {version: 0 for version in self.versions}

    def select_model_version(self) -> str:
        """
        Select a model version randomly for A/B testing.
        Uses 50/50 random split.

        Returns:
            Selected model version (e.g., "v1" or "v2")
        """
        selected = random.choice(self.versions)
        self._selection_counts[selected] += 1
        logger.debug(f"Selected model version: {selected}")
        return selected

    def get_model_for_version(self, version: str) -> SentimentModel:
        """
        Get model instance for specified version.

        Args:
            version: Model version identifier

        Returns:
            SentimentModel instance for the version

        Raises:
            ValueError: If version is not in configured versions
        """
        if version not in self.versions:
            raise ValueError(
                f"Invalid model version: {version}. Available: {self.versions}"
            )

        return SentimentModel.get_model(version)

    def get_selection_stats(self) -> Dict[str, int]:
        """
        Get selection count statistics for each version.

        Returns:
            Dictionary mapping version to selection count
        """
        return self._selection_counts.copy()

    def get_distribution_percentage(self) -> Dict[str, float]:
        """
        Get traffic distribution as percentages.

        Returns:
            Dictionary mapping version to percentage of traffic
        """
        total = sum(self._selection_counts.values())
        if total == 0:
            return {version: 0.0 for version in self.versions}

        return {
            version: round((count / total) * 100, 2)
            for version, count in self._selection_counts.items()
        }


# Global router instance
ab_router = ABTestRouter()
