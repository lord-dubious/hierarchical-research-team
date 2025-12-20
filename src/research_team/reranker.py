"""FlashRank reranker for search result optimization.

This module provides a wrapper around FlashRank for CPU-based document reranking,
filtering top results to save context tokens when sending to the LLM.
"""

from __future__ import annotations

from research_team.models import SearchResult


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
        self._ranker = None

    def _get_ranker(self):
        """Lazy load the FlashRank model."""
        if self._ranker is None:
            try:
                from flashrank import Ranker

                self._ranker = Ranker(model_name=self.model_name)
            except ImportError:
                print("FlashRank not installed, using fallback scoring")
                self._ranker = "fallback"
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

        ranker = self._get_ranker()

        if ranker == "fallback":
            # Fallback: simple keyword matching score
            return self._fallback_rerank(query, results, top_k)

        try:
            from flashrank import RerankRequest

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
            rerank_request = RerankRequest(query=query, passages=passages)

            # Execute reranking
            reranked = ranker.rerank(rerank_request)

            # Build reranked results
            reranked_results = []
            for item in reranked[:top_k]:
                idx = item["id"]
                original = results[idx]
                reranked_results.append(
                    SearchResult(
                        title=original.title,
                        url=original.url,
                        content=original.content,
                        engine=original.engine,
                        score=item["score"],
                    )
                )

            return reranked_results

        except Exception as e:
            print(f"FlashRank reranking failed: {e}, using fallback")
            return self._fallback_rerank(query, results, top_k)

    def _fallback_rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
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

            scored_results.append(
                SearchResult(
                    title=result.title,
                    url=result.url,
                    content=result.content,
                    engine=result.engine,
                    score=score,
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
