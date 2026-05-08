"""Hierarchical Research Team - local multi-agent research experiment.

A portfolio project using:
- LangGraph for multi-agent orchestration
- SearXNG for self-hosted meta-search when available
- FlashRank for CPU-based document reranking when available
- Gemini for model-generated summaries

External search, reranking, and model failures are surfaced through degraded
metadata on results, tasks, reports, or client state fields.
"""

from research_team.agents import ResearchTeam, create_team
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
from research_team.reranker import Reranker, rerank_results
from research_team.search import SearXNGClient, create_mock_results

__version__ = "1.0.0"

__all__ = [
    # Main classes
    "ResearchTeam",
    "SearXNGClient",
    "Reranker",
    # Factory functions
    "create_team",
    "create_mock_results",
    "rerank_results",
    # Models
    "AgentRole",
    "SearchResult",
    "SearchQuery",
    "ResearchTask",
    "ResearchPlan",
    "ResearchReport",
    "ReportSection",
    "TeamState",
    # Version
    "__version__",
]
