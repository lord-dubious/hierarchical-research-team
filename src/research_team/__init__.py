"""Hierarchical Research Team - Multi-agent research with LangGraph.

A production-ready research automation system using:
- LangGraph for multi-agent orchestration
- SearXNG for self-hosted meta-search
- FlashRank for CPU-based document reranking
- Gemini for AI reasoning
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
