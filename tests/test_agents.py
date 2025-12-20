"""Tests for the LangGraph research team agents."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_team.agents import ResearchTeam, SupervisorDecision, create_team
from research_team.models import (
    AgentRole,
    ReportSection,
    ResearchPlan,
    ResearchReport,
    ResearchTask,
    SearchResult,
    TeamState,
)


class TestResearchTeam:
    """Tests for ResearchTeam class."""

    def test_init_defaults(self, monkeypatch):
        """Test ResearchTeam initialization with defaults."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()
            assert team.model_name == "gemini-2.5-flash"
            assert team.searxng_url == "http://localhost:8080"

    def test_init_custom_values(self, monkeypatch):
        """Test ResearchTeam initialization with custom values."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam(
                model_name="gemini-1.5-pro",
                searxng_url="http://custom:9090",
            )
            assert team.model_name == "gemini-1.5-pro"
            assert team.searxng_url == "http://custom:9090"

    def test_build_graph(self, monkeypatch):
        """Test that the LangGraph is built correctly."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()
            assert team.graph is not None


class TestSupervisorNode:
    """Tests for supervisor node behavior."""

    def test_supervisor_creates_plan_first_iteration(self, monkeypatch):
        """Test supervisor creates research plan on first iteration."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = """
            1. What is AI?
            2. How does AI work?
            3. What are AI applications?
            """
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm

            team = ResearchTeam()

            state = TeamState(query="What is AI?")
            result = team._supervisor_node(state)

            assert "plan" in result
            assert result["plan"] is not None

    def test_supervisor_respects_max_iterations(self, monkeypatch):
        """Test supervisor stops at max iterations."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            state = TeamState(query="Test", iteration=10, max_iterations=10)
            result = team._supervisor_node(state)

            assert result.get("error") == "Max iterations reached"

    def test_supervisor_routes_to_writer_when_findings_complete(
        self, monkeypatch, sample_research_plan
    ):
        """Test supervisor routes to writer when all findings are gathered."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            state = TeamState(
                query="Test",
                plan=sample_research_plan,
                findings=["Finding 1", "Finding 2", "Finding 3"],  # Matches sub_questions count
            )
            result = team._supervisor_node(state)

            assert result["current_agent"] == AgentRole.WRITER

    def test_supervisor_routes_to_researcher_when_more_research_needed(
        self, monkeypatch, sample_research_plan
    ):
        """Test supervisor routes to researcher when more findings needed."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            state = TeamState(
                query="Test",
                plan=sample_research_plan,
                findings=["Finding 1"],  # Less than sub_questions
            )
            result = team._supervisor_node(state)

            assert result["current_agent"] == AgentRole.RESEARCHER


class TestResearcherNode:
    """Tests for researcher node behavior."""

    def test_researcher_returns_error_without_plan(self, monkeypatch):
        """Test researcher returns error when no plan exists."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            state = TeamState(query="Test", plan=None)
            result = team._researcher_node(state)

            assert result.get("error") == "No research plan available"

    def test_researcher_executes_search(
        self, monkeypatch, sample_research_plan, sample_search_results
    ):
        """Test researcher executes search for sub-question."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "AI is used in many healthcare applications..."
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm

            team = ResearchTeam()

            # Mock search and rerank
            with patch.object(team, "_execute_search", return_value=sample_search_results):
                with patch(
                    "research_team.agents.rerank_results", return_value=sample_search_results[:3]
                ):
                    state = TeamState(
                        query="AI in healthcare",
                        plan=sample_research_plan,
                        findings=[],
                    )
                    result = team._researcher_node(state)

                    assert "findings" in result
                    assert len(result["findings"]) == 1
                    assert "tasks" in result


class TestWriterNode:
    """Tests for writer node behavior."""

    def test_writer_returns_error_without_findings(self, monkeypatch):
        """Test writer returns error when no findings exist."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            state = TeamState(query="Test", findings=[])
            result = team._writer_node(state)

            assert result.get("error") == "No findings to write about"

    def test_writer_generates_report(self, monkeypatch, sample_research_plan):
        """Test writer generates research report."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "This report examines AI in healthcare..."
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm

            team = ResearchTeam()

            state = TeamState(
                query="AI in healthcare",
                plan=sample_research_plan,
                findings=["Finding 1", "Finding 2", "Finding 3"],
                tasks=[],
            )
            result = team._writer_node(state)

            assert "report" in result
            assert isinstance(result["report"], ResearchReport)


class TestRouting:
    """Tests for routing logic."""

    def test_route_finish_when_report_exists(self, monkeypatch, sample_research_report):
        """Test routing to finish when report exists."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            state = TeamState(query="Test", report=sample_research_report)
            route = team._route_from_supervisor(state)

            assert route == "finish"

    def test_route_finish_on_error(self, monkeypatch):
        """Test routing to finish on error."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            state = TeamState(query="Test", error="Something went wrong")
            route = team._route_from_supervisor(state)

            assert route == "finish"

    def test_route_to_writer(self, monkeypatch):
        """Test routing to writer."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            state = TeamState(query="Test", current_agent=AgentRole.WRITER)
            route = team._route_from_supervisor(state)

            assert route == "writer"

    def test_route_to_researcher(self, monkeypatch):
        """Test routing to researcher."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            state = TeamState(query="Test", current_agent=AgentRole.RESEARCHER)
            route = team._route_from_supervisor(state)

            assert route == "researcher"


class TestHelperMethods:
    """Tests for helper methods."""

    def test_create_research_plan(self, monkeypatch):
        """Test creating research plan from LLM response."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = """
            Objective: Understand AI
            
            1. What is artificial intelligence?
            2. How does machine learning work?
            3. What are the applications of AI?
            
            Methodology: Search, analyze, synthesize
            """
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm

            team = ResearchTeam()
            plan = team._create_research_plan("What is AI?")

            assert isinstance(plan, ResearchPlan)
            assert len(plan.sub_questions) >= 3

    def test_create_research_plan_fallback(self, monkeypatch):
        """Test research plan fallback when parsing fails."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Unparseable response"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm

            team = ResearchTeam()
            plan = team._create_research_plan("AI")

            # Should use fallback questions
            assert len(plan.sub_questions) == 3
            assert "What is AI" in plan.sub_questions[0]

    def test_execute_search_with_available_searxng(self, monkeypatch, sample_search_results):
        """Test search execution with available SearXNG."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            # Mock SearXNG availability and search
            team.search_client.is_available = MagicMock(return_value=True)

            async def mock_search(*args, **kwargs):
                return sample_search_results

            with patch.object(team.search_client, "search", mock_search):
                results = team._execute_search("test query")
                assert len(results) == 5

    def test_execute_search_fallback_to_mock(self, monkeypatch):
        """Test search falls back to mock when SearXNG unavailable."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            # Mock SearXNG as unavailable
            team.search_client.is_available = MagicMock(return_value=False)

            results = team._execute_search("test query")
            assert len(results) > 0

    def test_synthesize_finding(self, monkeypatch, sample_search_results):
        """Test synthesizing finding from search results."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "AI is a field of computer science..."
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm

            team = ResearchTeam()
            finding = team._synthesize_finding("What is AI?", sample_search_results)

            assert finding == "AI is a field of computer science..."


class TestCreateTeam:
    """Tests for create_team factory function."""

    def test_create_team_default(self, monkeypatch):
        """Test create_team with defaults."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = create_team()
            assert isinstance(team, ResearchTeam)

    def test_create_team_custom(self, monkeypatch):
        """Test create_team with custom values."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = create_team(
                model_name="gemini-1.5-pro",
                searxng_url="http://custom:8080",
            )
            assert team.model_name == "gemini-1.5-pro"
            assert team.searxng_url == "http://custom:8080"


class TestSupervisorDecision:
    """Tests for SupervisorDecision model."""

    def test_supervisor_decision_researcher(self):
        """Test supervisor decision to route to researcher."""
        decision = SupervisorDecision(
            next_agent="researcher",
            task="Search for information about AI",
            reasoning="Need more information before writing",
        )
        assert decision.next_agent == "researcher"

    def test_supervisor_decision_writer(self):
        """Test supervisor decision to route to writer."""
        decision = SupervisorDecision(
            next_agent="writer",
            task="Write the final report",
            reasoning="All information gathered",
        )
        assert decision.next_agent == "writer"

    def test_supervisor_decision_finish(self):
        """Test supervisor decision to finish."""
        decision = SupervisorDecision(
            next_agent="finish",
            task="Complete research",
            reasoning="Report is ready",
        )
        assert decision.next_agent == "finish"


class TestAsyncResearch:
    """Tests for async research methods."""

    @pytest.mark.asyncio
    async def test_research_async(self, monkeypatch, sample_research_report):
        """Test async research method."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            # Mock the graph execution
            async def mock_stream(state):
                yield {"writer": {"report": sample_research_report}}

            team.graph.astream = mock_stream

            report = await team.research("Test query")
            assert isinstance(report, ResearchReport)

    def test_research_sync(self, monkeypatch, sample_research_report):
        """Test sync research method."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("research_team.agents.ChatGoogleGenerativeAI"):
            team = ResearchTeam()

            # Mock the async research
            async def mock_research(query):
                return sample_research_report

            team.research = mock_research

            report = team.research_sync("Test query")
            assert isinstance(report, ResearchReport)
