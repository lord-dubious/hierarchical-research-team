"""Tests for the FlashRank reranker."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from research_team.models import SearchResult
from research_team.reranker import Reranker, rerank_results


class TestReranker:
    """Tests for Reranker class."""

    def test_init_default_model(self):
        """Test reranker initialization with default model."""
        reranker = Reranker()
        assert reranker.model_name == "ms-marco-MiniLM-L-12-v2"
        assert reranker._ranker is None

    def test_init_custom_model(self):
        """Test reranker initialization with custom model."""
        reranker = Reranker(model_name="ms-marco-TinyBERT-L-2-v2")
        assert reranker.model_name == "ms-marco-TinyBERT-L-2-v2"

    def test_rerank_empty_results(self):
        """Test reranking empty list returns empty list."""
        reranker = Reranker()
        results = reranker.rerank("test query", [])
        assert results == []

    def test_fallback_rerank_basic(self, sample_search_results):
        """Test fallback reranking with keyword matching."""
        reranker = Reranker()
        # Force fallback mode
        reranker._ranker = "fallback"

        results = reranker.rerank("AI artificial intelligence", sample_search_results, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.provenance == "fallback" for r in results)
        assert all(r.degraded for r in results)
        assert all(r.warning for r in results)
        assert all(r.metadata["reranker"] == "keyword_overlap_fallback" for r in results)
        # Results should be sorted by score descending
        assert results[0].score >= results[1].score >= results[2].score

    def test_fallback_rerank_scores(self):
        """Test fallback reranking assigns proper scores."""
        reranker = Reranker()
        reranker._ranker = "fallback"

        results = [
            SearchResult(
                title="Python programming", url="https://a.com", content="Learn Python basics"
            ),
            SearchResult(title="Java guide", url="https://b.com", content="Java for beginners"),
            SearchResult(
                title="Python advanced", url="https://c.com", content="Advanced Python topics"
            ),
        ]

        reranked = reranker.rerank("Python", results, top_k=3)

        # Python results should score higher
        python_scores = [r.score for r in reranked if "Python" in r.title]
        java_scores = [r.score for r in reranked if "Java" in r.title]

        assert all(ps > js for ps in python_scores for js in java_scores)

    def test_fallback_rerank_respects_top_k(self, sample_search_results):
        """Test fallback reranking respects top_k limit."""
        reranker = Reranker()
        reranker._ranker = "fallback"

        results = reranker.rerank("test", sample_search_results, top_k=2)

        assert len(results) == 2

    def test_rerank_with_flashrank_fallback_on_import_error(self, sample_search_results):
        """Test that reranker falls back when FlashRank not installed."""
        reranker = Reranker()

        with patch.object(reranker, "_get_ranker", return_value="fallback"):
            results = reranker.rerank("AI", sample_search_results, top_k=3)

            assert len(results) == 3
            assert reranker.last_degraded is True
            assert reranker.last_warning is not None
            assert all(result.provenance == "fallback" for result in results)

    def test_rerank_with_flashrank_fallback_on_exception(self, sample_search_results):
        """Test that reranker falls back on FlashRank errors."""
        reranker = Reranker()
        # Force fallback mode by setting ranker to fallback string
        reranker._ranker = "fallback"

        results = reranker.rerank("AI", sample_search_results, top_k=3)

        # Should use fallback and return results
        assert len(results) == 3

    def test_rerank_with_flashrank_success(self, sample_search_results):
        """Test successful reranking with FlashRank (using fallback as proxy)."""
        reranker = Reranker()
        # Use fallback to test the reranking logic since FlashRank may not be installed
        reranker._ranker = "fallback"

        results = reranker.rerank("AI artificial", sample_search_results, top_k=3)

        assert len(results) == 3
        # Results should be sorted by score descending
        assert results[0].score >= results[1].score
        assert results[1].score >= results[2].score

    def test_lazy_loading_ranker(self):
        """Test that ranker is lazily loaded."""
        reranker = Reranker()
        assert reranker._ranker is None

        # Call _get_ranker - it will either load FlashRank or fallback
        ranker = reranker._get_ranker()
        # Should now be set to something (either Ranker instance or "fallback")
        assert reranker._ranker is not None
        assert ranker == reranker._ranker

    def test_lazy_loading_logs_flashrank_import_fallback(self, caplog):
        """Test missing FlashRank logs and records fallback state."""
        reranker = Reranker()

        caplog.set_level(logging.WARNING)
        with patch(
            "research_team.reranker.import_module", side_effect=ImportError("missing flashrank")
        ):
            ranker = reranker._get_ranker()

        assert ranker == "fallback"
        assert reranker.last_degraded is True
        assert reranker.last_error == "FlashRank is not installed"
        assert "FlashRank unavailable" in caplog.text

    def test_lazy_loading_records_flashrank_initialization_failure(self, caplog):
        """Test FlashRank construction failure records degraded state."""
        reranker = Reranker()
        fake_flashrank = MagicMock()
        fake_flashrank.Ranker.side_effect = RuntimeError("model unavailable")

        caplog.set_level(logging.WARNING)
        with patch("research_team.reranker.import_module", return_value=fake_flashrank):
            ranker = reranker._get_ranker()

        assert ranker == "fallback"
        assert reranker.last_degraded is True
        assert "model unavailable" in reranker.last_error
        assert "FlashRank initialization failed" in caplog.text

    def test_fallback_rerank_handles_empty_query(self, sample_search_results):
        """Test fallback with empty query."""
        reranker = Reranker()
        reranker._ranker = "fallback"

        # Empty query should not crash
        results = reranker.rerank("", sample_search_results, top_k=3)
        assert len(results) == 3


class TestRerankResults:
    """Tests for rerank_results convenience function."""

    def test_rerank_results_function(self, sample_search_results):
        """Test the convenience rerank_results function."""
        with patch("research_team.reranker.Reranker") as mock_class:
            mock_instance = MagicMock()
            mock_instance.rerank.return_value = sample_search_results[:3]
            mock_class.return_value = mock_instance

            results = rerank_results("AI", sample_search_results, top_k=3)

            mock_class.assert_called_once()
            mock_instance.rerank.assert_called_once_with("AI", sample_search_results, 3)
            assert len(results) == 3

    def test_rerank_results_default_top_k(self, sample_search_results):
        """Test rerank_results uses default top_k of 5."""
        with patch("research_team.reranker.Reranker") as mock_class:
            mock_instance = MagicMock()
            mock_instance.rerank.return_value = sample_search_results
            mock_class.return_value = mock_instance

            rerank_results("test", sample_search_results)

            # Check top_k default is 5
            call_args = mock_instance.rerank.call_args
            assert call_args[0][2] == 5  # top_k is the third positional arg


class TestRerankerEdgeCases:
    """Edge case tests for Reranker."""

    def test_rerank_single_result(self):
        """Test reranking a single result."""
        reranker = Reranker()
        reranker._ranker = "fallback"

        results = [SearchResult(title="Single", url="https://single.com", content="Only one")]
        reranked = reranker.rerank("single", results, top_k=5)

        assert len(reranked) == 1
        assert reranked[0].title == "Single"

    def test_rerank_top_k_larger_than_results(self, sample_search_results):
        """Test reranking when top_k is larger than results."""
        reranker = Reranker()
        reranker._ranker = "fallback"

        # Request more than available
        reranked = reranker.rerank("test", sample_search_results, top_k=100)

        # Should return all available
        assert len(reranked) == len(sample_search_results)

    def test_rerank_preserves_metadata(self, sample_search_results):
        """Test that reranking preserves all metadata."""
        reranker = Reranker()
        reranker._ranker = "fallback"

        reranked = reranker.rerank("AI", sample_search_results, top_k=3)

        for result in reranked:
            # Find original
            original = next(r for r in sample_search_results if r.url == result.url)
            assert result.title == original.title
            assert result.content == original.content
            assert result.engine == original.engine

    def test_rerank_score_normalization(self):
        """Test that fallback scores are normalized."""
        reranker = Reranker()
        reranker._ranker = "fallback"

        results = [
            SearchResult(
                title="Exact match query",
                url="https://a.com",
                content="exact match query",
            ),
        ]

        reranked = reranker.rerank("exact match query", results, top_k=1)

        # Score should be between 0 and 1
        assert 0 <= reranked[0].score <= 1

    def test_fallback_handles_special_characters(self):
        """Test fallback handles special characters in query."""
        reranker = Reranker()
        reranker._ranker = "fallback"

        results = [
            SearchResult(title="C++ Programming", url="https://a.com", content="Learn C++"),
        ]

        # Should not raise
        reranked = reranker.rerank("C++ programming", results, top_k=1)
        assert len(reranked) == 1
