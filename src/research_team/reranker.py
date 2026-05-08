"""FlashRank reranker for search result optimization.

This module provides a wrapper around FlashRank for CPU-based document reranking,
filtering top results to save context tokens when sending to the LLM.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Literal, Protocol

from research_team.models import SearchResult

logger = logging.getLogger(__name__)


class _RankerProtocol(Protocol):
    """Minimal FlashRank interface used by this wrapper."""

    def rerank(self, request: Any) -> list[dict[str, Any]]:
        """Rerank passages for a FlashRank request."""
        ...


class Reranker:
    """FlashRank-based document reranker.

    FlashRank is an ultra-lite (4MB) model that runs on CPU,
    eliminating the need for heavy GPUs for document reranking.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """Initialize the reranker.

        Args:
            model_name: FlashRank model to use. Options include:
                - ms-marco-MiniLM-L-12-v2 (default, best balance)
                - ms-marco-TinyBERT-L-2-v2 (fastest, smaller)
        """
        self.model_name = model_name
        self._ranker: _RankerProtocol | Literal["fallback"] | None = None
        self.last_error: str | None = None
        self.last_warning: str | None = None
        self.last_degraded: bool = False

    def _get_ranker(self) -> _RankerProtocol | Literal["fallback"]:
        """Lazy load the FlashRank model."""
        if self._ranker is None:
            try:
                flashrank = import_module("flashrank")
                ranker_cls: Any = flashrank.Ranker
                self._ranker = ranker_cls(model_name=self.model_name)
            except ImportError:
                self.last_error = "FlashRank is not installed"
                self.last_warning = (
                    "FlashRank unavailable; using keyword-overlap fallback reranking"
                )
                self.last_degraded = True
                logger.warning(self.last_warning)
                self._ranker = "fallback"
            except Exception as exc:
                self.last_error = f"FlashRank initialization failed: {exc}"
                self.last_warning = (
                    "FlashRank initialization degraded; using keyword-overlap fallback reranking"
                )
                self.last_degraded = True
                logger.warning(self.last_error)
                self._ranker = "fallback"
        assert self._ranker is not None
        return self._ranker

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank search results by relevance to query.

        Args:
            query: The search query.
            results: List of search results to rerank.
            top_k: Number of top results to return.

        Returns:
            List of reranked SearchResult objects with updated scores.
        """
        if not results:
            return []

        self.last_error = None
        self.last_warning = None
        self.last_degraded = False

        ranker = self._get_ranker()

        if ranker == "fallback":
            if not self.last_warning:
                self.last_warning = (
                    "FlashRank unavailable; using keyword-overlap fallback reranking"
                )
            self.last_degraded = True
            # Fallback: simple keyword matching score
            return self._fallback_rerank(query, results, top_k)

        try:
            flashrank = import_module("flashrank")
            rerank_request_cls: Any = flashrank.RerankRequest

            # Prepare passages for FlashRank
            passages = []
            for i, result in enumerate(results):
                passages.append(
                    {
                        "id": i,
                        "text": f"{result.title}. {result.content}",
                        "meta": {"url": result.url, "engine": result.engine},
                    }
                )

            # Create rerank request
            rerank_request = rerank_request_cls(query=query, passages=passages)

            # Execute reranking
            reranked = ranker.rerank(rerank_request)

            # Build reranked results
            reranked_results = []
            for item in reranked[:top_k]:
                idx = item["id"]
                original = results[idx]
                reranked_results.append(
                    original.model_copy(
                        update={
                            "score": item["score"],
                            "provenance": "flashrank",
                            "metadata": {
                                **original.metadata,
                                "reranker": "flashrank",
                                "reranker_model": self.model_name,
                            },
                        }
                    )
                )

            return reranked_results

        except Exception as e:
            self.last_error = f"FlashRank reranking failed: {e}"
            self.last_warning = (
                "FlashRank reranking degraded; using keyword-overlap fallback reranking"
            )
            self.last_degraded = True
            logger.warning(self.last_error)
            return self._fallback_rerank(query, results, top_k)

    def _fallback_rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
        reason: str | None = None,
    ) -> list[SearchResult]:
        """Fallback reranking using simple keyword matching.

        Args:
            query: Search query.
            results: Search results.
            top_k: Number of results to return.

        Returns:
            Reranked results with simple scores.
        """
        query_terms = set(query.lower().split())

        scored_results = []
        for result in results:
            text = f"{result.title} {result.content}".lower()
            text_terms = set(text.split())

            # Simple Jaccard-like overlap score
            overlap = len(query_terms & text_terms)
            score = overlap / max(len(query_terms), 1)

            warning = reason or self.last_warning or "Keyword-overlap fallback reranking used"
            scored_results.append(
                result.model_copy(
                    update={
                        "score": score,
                        "provenance": "fallback",
                        "degraded": True,
                        "warning": warning,
                        "error": self.last_error or result.error,
                        "metadata": {
                            **result.metadata,
                            "reranker": "keyword_overlap_fallback",
                            "reranker_model": self.model_name,
                            "reranker_degraded": True,
                        },
                    }
                )
            )

        # Sort by score descending
        scored_results.sort(key=lambda x: x.score, reverse=True)

        return scored_results[:top_k]


def rerank_results(
    query: str,
    results: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """Convenience function for reranking search results.

    Args:
        query: The search query.
        results: List of search results.
        top_k: Number of top results to return.

    Returns:
        Reranked results.
    """
    reranker = Reranker()
    return reranker.rerank(query, results, top_k)
