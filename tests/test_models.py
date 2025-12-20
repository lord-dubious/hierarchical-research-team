"""Tests for Pydantic models."""

from __future__ import annotations

from datetime import datetime

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


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_search_result(self):
        """Test creating a search result."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            content="Test content",
            engine="google",
            score=0.85,
        )
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.content == "Test content"
        assert result.engine == "google"
        assert result.score == 0.85

    def test_search_result_defaults(self):
        """Test default values for search result."""
        result = SearchResult(
            title="Test",
            url="https://example.com",
        )
        assert result.content == ""
        assert result.engine == ""
        assert result.score == 0.0

    def test_search_result_extra_fields_ignored(self):
        """Test that extra fields are ignored."""
        result = SearchResult(
            title="Test",
            url="https://example.com",
            extra_field="ignored",
        )
        assert not hasattr(result, "extra_field")


class TestSearchQuery:
    """Tests for SearchQuery model."""

    def test_create_search_query(self):
        """Test creating a search query."""
        query = SearchQuery(
            query="test query",
            num_results=20,
            categories=["general", "news"],
        )
        assert query.query == "test query"
        assert query.num_results == 20
        assert query.categories == ["general", "news"]

    def test_search_query_defaults(self):
        """Test default values for search query."""
        query = SearchQuery(query="test")
        assert query.num_results == 10
        assert query.categories == ["general"]


class TestAgentRole:
    """Tests for AgentRole enum."""

    def test_agent_roles(self):
        """Test agent role values."""
        assert AgentRole.SUPERVISOR.value == "supervisor"
        assert AgentRole.RESEARCHER.value == "researcher"
        assert AgentRole.WRITER.value == "writer"
        assert AgentRole.CHART_GENERATOR.value == "chart_generator"

    def test_agent_role_from_string(self):
        """Test creating agent role from string."""
        role = AgentRole("supervisor")
        assert role == AgentRole.SUPERVISOR


class TestResearchTask:
    """Tests for ResearchTask model."""

    def test_create_research_task(self):
        """Test creating a research task."""
        task = ResearchTask(
            task_id="task-001",
            query="What is AI?",
            assigned_to=AgentRole.RESEARCHER,
        )
        assert task.task_id == "task-001"
        assert task.query == "What is AI?"
        assert task.assigned_to == AgentRole.RESEARCHER
        assert task.status == "pending"

    def test_research_task_with_result(self):
        """Test research task with result."""
        task = ResearchTask(
            task_id="task-001",
            query="What is AI?",
            assigned_to=AgentRole.RESEARCHER,
            status="completed",
            result="AI is artificial intelligence...",
        )
        assert task.status == "completed"
        assert task.result == "AI is artificial intelligence..."

    def test_research_task_with_sources(self, sample_search_results):
        """Test research task with sources."""
        task = ResearchTask(
            task_id="task-001",
            query="What is AI?",
            assigned_to=AgentRole.RESEARCHER,
            sources=sample_search_results[:2],
        )
        assert len(task.sources) == 2
        assert task.sources[0].title == "Understanding AI - Wikipedia"


class TestResearchPlan:
    """Tests for ResearchPlan model."""

    def test_create_research_plan(self):
        """Test creating a research plan."""
        plan = ResearchPlan(
            objective="Understand AI",
            sub_questions=["What is AI?", "How does AI work?"],
            methodology=["Search", "Analyze"],
            expected_output="Report",
        )
        assert plan.objective == "Understand AI"
        assert len(plan.sub_questions) == 2
        assert len(plan.methodology) == 2
        assert plan.expected_output == "Report"


class TestReportSection:
    """Tests for ReportSection model."""

    def test_create_report_section(self):
        """Test creating a report section."""
        section = ReportSection(
            heading="Introduction",
            content="This is the introduction...",
            sources=["https://example.com"],
        )
        assert section.heading == "Introduction"
        assert section.content == "This is the introduction..."
        assert section.sources == ["https://example.com"]

    def test_report_section_empty_sources(self):
        """Test report section with no sources."""
        section = ReportSection(
            heading="Conclusion",
            content="Summary of findings...",
        )
        assert section.sources == []


class TestResearchReport:
    """Tests for ResearchReport model."""

    def test_create_research_report(self, sample_search_results):
        """Test creating a research report."""
        sections = [
            ReportSection(heading="Section 1", content="Content 1"),
            ReportSection(heading="Section 2", content="Content 2"),
        ]
        report = ResearchReport(
            title="Test Report",
            summary="This is a summary",
            sections=sections,
            sources=sample_search_results[:2],
        )
        assert report.title == "Test Report"
        assert report.summary == "This is a summary"
        assert len(report.sections) == 2
        assert len(report.sources) == 2
        assert isinstance(report.generated_at, datetime)

    def test_research_report_with_timestamp(self):
        """Test research report with specific timestamp."""
        timestamp = datetime(2024, 6, 15, 10, 30, 0)
        report = ResearchReport(
            title="Report",
            summary="Summary",
            sections=[],
            sources=[],
            generated_at=timestamp,
        )
        assert report.generated_at == timestamp


class TestTeamState:
    """Tests for TeamState model."""

    def test_create_team_state(self):
        """Test creating team state."""
        state = TeamState(query="What is AI?")
        assert state.query == "What is AI?"
        assert state.plan is None
        assert state.tasks == []
        assert state.search_results == []
        assert state.findings == []
        assert state.report is None
        assert state.current_agent == AgentRole.SUPERVISOR
        assert state.iteration == 0
        assert state.max_iterations == 10
        assert state.error is None

    def test_team_state_with_plan(self, sample_research_plan):
        """Test team state with plan."""
        state = TeamState(
            query="What is AI?",
            plan=sample_research_plan,
        )
        assert state.plan is not None
        assert state.plan.objective == sample_research_plan.objective

    def test_team_state_iteration_tracking(self):
        """Test team state iteration tracking."""
        state = TeamState(
            query="Test",
            iteration=5,
            max_iterations=10,
        )
        assert state.iteration == 5
        assert state.max_iterations == 10

    def test_team_state_with_error(self):
        """Test team state with error."""
        state = TeamState(
            query="Test",
            error="Something went wrong",
        )
        assert state.error == "Something went wrong"


class TestModelIntegration:
    """Integration tests for models working together."""

    def test_full_research_workflow_models(self, sample_search_results):
        """Test models through a full workflow."""
        # Create search query
        query = SearchQuery(query="AI research")

        # Create plan
        plan = ResearchPlan(
            objective="Research AI",
            sub_questions=["What is AI?"],
            methodology=["Search"],
            expected_output="Report",
        )

        # Create task
        task = ResearchTask(
            task_id="task-001",
            query="What is AI?",
            assigned_to=AgentRole.RESEARCHER,
            status="completed",
            result="AI is...",
            sources=sample_search_results[:2],
        )

        # Create section
        section = ReportSection(
            heading=task.query,
            content=task.result,
            sources=[s.url for s in task.sources],
        )

        # Create report
        report = ResearchReport(
            title="AI Research Report",
            summary="Summary of AI research",
            sections=[section],
            sources=task.sources,
        )

        # Create final state
        state = TeamState(
            query=query.query,
            plan=plan,
            tasks=[task],
            search_results=sample_search_results[:2],
            findings=[task.result],
            report=report,
            current_agent=AgentRole.WRITER,
            iteration=5,
        )

        assert state.report is not None
        assert state.report.title == "AI Research Report"
        assert len(state.tasks) == 1
        assert state.tasks[0].status == "completed"
