"""Pydantic models for the Hierarchical Research Team."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Roles for agents in the research team."""

    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    WRITER = "writer"
    CHART_GENERATOR = "chart_generator"


class SearchResult(BaseModel):
    """A single search result from SearXNG."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the result")
    content: str = Field(default="", description="Snippet/description")
    engine: str = Field(default="", description="Search engine source")
    score: float = Field(default=0.0, description="Relevance score after reranking")

    model_config = {"extra": "ignore"}


class SearchQuery(BaseModel):
    """A search query to execute."""

    query: str = Field(description="The search query string")
    num_results: int = Field(default=10, description="Number of results to fetch")
    categories: list[str] = Field(
        default_factory=lambda: ["general"], description="SearXNG categories"
    )


class ResearchTask(BaseModel):
    """A research task assigned to an agent."""

    task_id: str = Field(description="Unique task identifier")
    query: str = Field(description="The research question or topic")
    context: str = Field(default="", description="Additional context for the task")
    assigned_to: AgentRole = Field(description="Agent role assigned to this task")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(default="pending")
    result: str | None = Field(default=None, description="Task result when completed")
    sources: list[SearchResult] = Field(default_factory=list, description="Sources used")
    created_at: datetime = Field(default_factory=datetime.now)


class ResearchPlan(BaseModel):
    """A structured research plan created by the supervisor."""

    objective: str = Field(description="Main research objective")
    sub_questions: list[str] = Field(description="Sub-questions to answer")
    methodology: list[str] = Field(description="Steps to complete the research")
    expected_output: str = Field(description="What the final output should look like")


class ResearchReport(BaseModel):
    """Final research report synthesized from all findings."""

    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary")
    sections: list[ReportSection] = Field(description="Report sections")
    sources: list[SearchResult] = Field(description="All sources used")
    generated_at: datetime = Field(default_factory=datetime.now)


class ReportSection(BaseModel):
    """A section of the research report."""

    heading: str = Field(description="Section heading")
    content: str = Field(description="Section content")
    sources: list[str] = Field(default_factory=list, description="Source URLs for this section")


# Fix forward reference
ResearchReport.model_rebuild()


class TeamState(BaseModel):
    """State for the LangGraph research team workflow."""

    query: str = Field(description="Original research query")
    plan: ResearchPlan | None = Field(default=None, description="Research plan")
    tasks: list[ResearchTask] = Field(default_factory=list, description="All tasks")
    search_results: list[SearchResult] = Field(default_factory=list, description="Search results")
    findings: list[str] = Field(default_factory=list, description="Research findings")
    report: ResearchReport | None = Field(default=None, description="Final report")
    current_agent: AgentRole = Field(default=AgentRole.SUPERVISOR)
    iteration: int = Field(default=0, description="Current iteration count")
    max_iterations: int = Field(default=10, description="Max iterations to prevent loops")
    error: str | None = Field(default=None, description="Error message if failed")

    model_config = {"arbitrary_types_allowed": True}
