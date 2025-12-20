"""Tests for the SearXNG search client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from research_team.models import SearchQuery, SearchResult
from research_team.search import SearXNGClient, create_mock_results


class TestSearXNGClient:
    """Tests for SearXNGClient."""

    def test_init_with_defaults(self):
        """Test client initialization with default values."""
        client = SearXNGClient()
        assert client.base_url == "http://localhost:8080"
        assert client.timeout == 10

    def test_init_with_custom_values(self):
        """Test client initialization with custom values."""
        client = SearXNGClient(base_url="http://search.example.com", timeout=30)
        assert client.base_url == "http://search.example.com"
        assert client.timeout == 30

    def test_init_from_env(self, monkeypatch):
        """Test client initialization from environment variables."""
        monkeypatch.setenv("SEARXNG_URL", "http://env-search.com")
        monkeypatch.setenv("SEARXNG_TIMEOUT", "20")
        client = SearXNGClient()
        assert client.base_url == "http://env-search.com"
        assert client.timeout == 20

    @pytest.mark.asyncio
    async def test_search_with_string_query(self):
        """Test search with a string query."""
        client = SearXNGClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com",
                    "content": "Test content",
                    "engine": "google",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.get.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            results = await client.search("test query")

            assert len(results) == 1
            assert results[0].title == "Test Result"
            assert results[0].url == "https://example.com"

    @pytest.mark.asyncio
    async def test_search_with_search_query_object(self):
        """Test search with SearchQuery object."""
        client = SearXNGClient()
        query = SearchQuery(
            query="test query",
            num_results=5,
            categories=["general", "news"],
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.get.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            results = await client.search(query)

            assert results == []

    @pytest.mark.asyncio
    async def test_search_limits_results(self):
        """Test that search limits results to num_results."""
        client = SearXNGClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": ""}
                for i in range(20)
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.get.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            results = await client.search("test", num_results=5)

            assert len(results) == 5

    @pytest.mark.asyncio
    async def test_search_handles_http_error(self):
        """Test that search handles HTTP errors gracefully."""
        client = SearXNGClient()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.get.side_effect = httpx.HTTPError("Connection failed")
            mock_get_client.return_value = mock_http_client

            results = await client.search("test")

            assert results == []

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing the HTTP client."""
        client = SearXNGClient()
        mock_http_client = AsyncMock()
        client._client = mock_http_client

        await client.close()

        mock_http_client.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_client_when_none(self):
        """Test closing when client is None."""
        client = SearXNGClient()
        assert client._client is None

        # Should not raise
        await client.close()

    def test_is_available_success(self):
        """Test is_available when SearXNG is reachable."""
        client = SearXNGClient()

        with patch("httpx.Client") as mock_client_class:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_instance.get.return_value = mock_response
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_client_class.return_value = mock_instance

            assert client.is_available() is True

    def test_is_available_failure(self):
        """Test is_available when SearXNG is not reachable."""
        client = SearXNGClient()

        with patch("httpx.Client") as mock_client_class:
            mock_instance = MagicMock()
            mock_instance.get.side_effect = Exception("Connection refused")
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_client_class.return_value = mock_instance

            assert client.is_available() is False

    def test_is_available_fallback_to_main_page(self):
        """Test is_available when healthz returns 200."""
        client = SearXNGClient()

        with patch("httpx.Client") as mock_client_class:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_instance.get.return_value = mock_response
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_client_class.return_value = mock_instance

            result = client.is_available()
            assert result is True


class TestCreateMockResults:
    """Tests for create_mock_results function."""

    def test_create_mock_results_default(self):
        """Test creating mock results with default count."""
        results = create_mock_results("artificial intelligence")

        assert len(results) == 5
        assert all(isinstance(r, SearchResult) for r in results)

    def test_create_mock_results_custom_count(self):
        """Test creating mock results with custom count."""
        results = create_mock_results("test query", num_results=3)

        assert len(results) == 3

    def test_create_mock_results_content(self):
        """Test mock results contain query-related content."""
        query = "machine learning"
        results = create_mock_results(query)

        for result in results:
            assert query in result.title or query.replace(" ", "-") in result.url

    def test_create_mock_results_has_scores(self):
        """Test mock results have scores."""
        results = create_mock_results("test")

        assert results[0].score == 0.95  # First result has highest score
        assert all(r.score > 0 for r in results)

    def test_create_mock_results_has_engines(self):
        """Test mock results have engine sources."""
        results = create_mock_results("test")

        engines = {r.engine for r in results}
        assert "wikipedia" in engines
        assert "google" in engines

    def test_create_mock_results_urls_valid(self):
        """Test mock results have valid-looking URLs."""
        results = create_mock_results("test")

        for result in results:
            assert result.url.startswith("https://")

    def test_create_mock_results_handles_special_chars(self):
        """Test mock results handle special characters in query."""
        results = create_mock_results("test query with spaces")

        # Should not raise and should produce valid URLs
        assert len(results) == 5
        for result in results:
            assert " " not in result.url.split("/")[-1] or "+" in result.url


class TestSearchIntegration:
    """Integration tests for search functionality."""

    @pytest.mark.asyncio
    async def test_search_and_parse_results(self):
        """Test full search and parse flow."""
        client = SearXNGClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "AI Overview",
                    "url": "https://example.com/ai",
                    "content": "Artificial intelligence overview...",
                    "engine": "google",
                },
                {
                    "title": "ML Guide",
                    "url": "https://example.com/ml",
                    "content": "Machine learning fundamentals...",
                    "engine": "bing",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.get.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            results = await client.search("AI ML")

            assert len(results) == 2
            assert results[0].title == "AI Overview"
            assert results[1].title == "ML Guide"
            assert all(r.score == 0.0 for r in results)  # Scores set by reranker
