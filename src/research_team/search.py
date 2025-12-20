"""SearXNG client for self-hosted meta-search.

This module provides a client for interacting with a self-hosted SearXNG instance,
replacing expensive search APIs like Tavily or Serper.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

from research_team.models import SearchQuery, SearchResult

load_dotenv()


class SearXNGClient:
    """Client for SearXNG meta-search engine.

    SearXNG aggregates results from multiple search engines (Google, Bing, DuckDuckGo)
    without per-query costs.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
    ):
        """Initialize SearXNG client.

        Args:
            base_url: SearXNG instance URL. Defaults to SEARXNG_URL env var.
            timeout: Request timeout in seconds. Defaults to SEARXNG_TIMEOUT env var.
        """
        self.base_url = base_url or os.getenv("SEARXNG_URL", "http://localhost:8080")
        self.timeout = timeout or int(os.getenv("SEARXNG_TIMEOUT", "10"))
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search(
        self,
        query: str | SearchQuery,
        num_results: int = 10,
        categories: list[str] | None = None,
    ) -> list[SearchResult]:
        """Execute a search query against SearXNG.

        Args:
            query: Search query string or SearchQuery object.
            num_results: Maximum number of results to return.
            categories: SearXNG categories (general, images, news, etc.).

        Returns:
            List of SearchResult objects.
        """
        if isinstance(query, SearchQuery):
            search_query = query.query
            num_results = query.num_results
            categories = query.categories
        else:
            search_query = query
            categories = categories or ["general"]

        client = await self._get_client()

        params: dict[str, Any] = {
            "q": search_query,
            "format": "json",
            "categories": ",".join(categories),
        }

        try:
            response = await client.get(f"{self.base_url}/search", params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", [])[:num_results]:
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                        engine=item.get("engine", ""),
                        score=0.0,  # Will be set by reranker
                    )
                )

            return results

        except httpx.HTTPError as e:
            # Return empty results on error, log for debugging
            print(f"SearXNG search error: {e}")
            return []

    async def search_sync(
        self,
        query: str,
        num_results: int = 10,
        categories: list[str] | None = None,
    ) -> list[SearchResult]:
        """Synchronous search wrapper."""
        return await self.search(query, num_results, categories)

    def is_available(self) -> bool:
        """Check if SearXNG is available.

        Returns:
            True if SearXNG responds, False otherwise.
        """
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.base_url}/healthz")
                return response.status_code == 200
        except Exception:
            # Try the main page as fallback
            try:
                with httpx.Client(timeout=5) as client:
                    response = client.get(self.base_url)
                    return response.status_code == 200
            except Exception:
                return False


def create_mock_results(query: str, num_results: int = 5) -> list[SearchResult]:
    """Create mock search results for testing when SearXNG is unavailable.

    Args:
        query: Search query.
        num_results: Number of mock results.

    Returns:
        List of mock SearchResult objects.
    """
    mock_results = [
        SearchResult(
            title=f"Understanding {query} - Wikipedia",
            url=f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            content=f"A comprehensive overview of {query} and its applications...",
            engine="wikipedia",
            score=0.95,
        ),
        SearchResult(
            title=f"{query}: A Complete Guide",
            url=f"https://example.com/guides/{query.replace(' ', '-')}",
            content=f"This guide covers everything you need to know about {query}...",
            engine="google",
            score=0.88,
        ),
        SearchResult(
            title=f"Latest Research on {query}",
            url=f"https://arxiv.org/search/?query={query.replace(' ', '+')}",
            content=f"Recent academic papers and research findings on {query}...",
            engine="arxiv",
            score=0.82,
        ),
        SearchResult(
            title=f"{query} Best Practices",
            url=f"https://medium.com/tags/{query.replace(' ', '-')}",
            content=f"Expert opinions and best practices for implementing {query}...",
            engine="bing",
            score=0.75,
        ),
        SearchResult(
            title=f"GitHub - {query} Examples",
            url=f"https://github.com/topics/{query.replace(' ', '-')}",
            content=f"Open source projects and examples related to {query}...",
            engine="github",
            score=0.70,
        ),
    ]

    return mock_results[:num_results]
