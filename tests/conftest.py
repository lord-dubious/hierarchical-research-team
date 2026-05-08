"""Test fixtures for the Hierarchical Research Team."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from research_team.models import (
    AgentRole,
    ReportSection,
    ResearchPlan,
    ResearchReport,
    ResearchTask,
    SearchQuery,
    SearchResult,
    TeamState,
)


@pytest.fixture
def sample_search_result() -> SearchResult:
    """Create a sample search result."""
    return SearchResult(
        title="Sample Article Title",
        url="https://example.com/article",
        content="This is a sample article about AI and machine learning.",
        engine="google",
        score=0.85,
    )


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create a list of sample search results."""
    return [
        SearchResult(
            title="Understanding AI - Wikipedia",
            url="https://en.wikipedia.org/wiki/Artificial_intelligence",
            content="Artificial intelligence is the simulation of human intelligence...",
            engine="wikipedia",
            score=0.95,
        ),
        SearchResult(
            title="AI Complete Guide",
            url="https://example.com/ai-guide",
            content="A comprehensive guide to artificial intelligence...",
            engine="google",
            score=0.88,
        ),
        SearchResult(
            title="Latest AI Research",
            url="https://arxiv.org/ai",
            content="Recent advances in deep learning and neural networks...",
            engine="arxiv",
            score=0.82,
        ),
        SearchResult(
            title="Machine Learning Best Practices",
            url="https://medium.com/ml-practices",
            content="Expert insights on implementing machine learning...",
            engine="bing",
            score=0.75,
        ),
        SearchResult(
            title="AI on GitHub",
            url="https://github.com/topics/ai",
            content="Open source AI projects and implementations...",
            engine="github",
            score=0.70,
        ),
    ]


@pytest.fixture
def sample_search_query() -> SearchQuery:
    """Create a sample search query."""
    return SearchQuery(
        query="What is artificial intelligence?",
        num_results=10,
        categories=["general"],
    )


@pytest.fixture
def sample_research_task() -> ResearchTask:
    """Create a sample research task."""
    return ResearchTask(
        task_id="task-12345678",
        query="What are the main applications of AI?",
        context="Focus on recent developments in 2024",
        assigned_to=AgentRole.RESEARCHER,
        status="pending",
    )


@pytest.fixture
def sample_research_plan() -> ResearchPlan:
    """Create a sample research plan."""
    return ResearchPlan(
        objective="Understand the impact of AI on healthcare",
        sub_questions=[
            "What are the current applications of AI in healthcare?",
            "How is AI improving diagnostic accuracy?",
            "What are the ethical considerations of AI in medicine?",
        ],
        methodology=["Search for information", "Analyze results", "Synthesize findings"],
        expected_output="Comprehensive research report on AI in healthcare",
    )


@pytest.fixture
def sample_report_section() -> ReportSection:
    """Create a sample report section."""
    return ReportSection(
        heading="Applications of AI in Healthcare",
        content="AI is transforming healthcare through improved diagnostics...",
        sources=["https://example.com/ai-healthcare", "https://medical.ai/overview"],
    )


@pytest.fixture
def sample_research_report(
    sample_report_section: ReportSection,
    sample_search_results: list[SearchResult],
) -> ResearchReport:
    """Create a sample research report."""
    return ResearchReport(
        title="Research Report: AI in Healthcare",
        summary="This report examines the impact of artificial intelligence on healthcare...",
        sections=[sample_report_section],
        sources=sample_search_results[:3],
        generated_at=datetime(2024, 1, 15, 10, 30, 0),
    )


@pytest.fixture
def sample_team_state(sample_research_plan: ResearchPlan) -> TeamState:
    """Create a sample team state."""
    return TeamState(
        query="What is the impact of AI on healthcare?",
        plan=sample_research_plan,
        tasks=[],
        search_results=[],
        findings=[],
        report=None,
        current_agent=AgentRole.SUPERVISOR,
        iteration=0,
        max_iterations=10,
    )


@pytest.fixture
def mock_gemini_response():
    """Create a mock Gemini API response."""
    mock_response = MagicMock()
    mock_response.content = """Here is a research plan:

1. What is artificial intelligence and how does it work?
2. What are the main applications of AI in various industries?
3. How is AI expected to evolve in the next decade?

Methodology:
- Search for relevant information
- Analyze and synthesize findings
- Generate comprehensive report

Expected output: A detailed research report."""
    return mock_response


@pytest.fixture
def mock_llm(mock_gemini_response):
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = mock_gemini_response
    return mock


@pytest.fixture
def mock_search_client(sample_search_results):
    """Create a mock SearXNG client."""
    mock = MagicMock()
    mock.is_available.return_value = True
    mock.search = AsyncMock(return_value=sample_search_results)
    return mock


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-flash")
