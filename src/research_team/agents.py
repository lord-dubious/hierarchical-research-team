"""LangGraph-based hierarchical research team.

This module implements a multi-agent research workflow using LangGraph,
with a supervisor coordinating researcher and writer agents.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from research_team.models import (
    AgentRole,
    ReportSection,
    ResearchPlan,
    ResearchReport,
    ResearchTask,
    SearchResult,
    TeamState,
)
from research_team.reranker import Reranker
from research_team.search import SearXNGClient, create_mock_results

load_dotenv()

logger = logging.getLogger(__name__)


class SupervisorDecision(BaseModel):
    """Structured decision from the supervisor."""

    next_agent: Literal["researcher", "writer", "finish"] = Field(
        description="Which agent to route to next"
    )
    task: str = Field(description="Task description for the next agent")
    reasoning: str = Field(description="Reasoning for this decision")


class ResearchTeam:
    """Hierarchical research team using LangGraph.

    Architecture:
    - Supervisor: Routes tasks, creates research plan, synthesizes results
    - Researcher: Executes searches and gathers information
    - Writer: Synthesizes findings into a coherent report
    """

    def __init__(
        self,
        model_name: str | None = None,
        searxng_url: str | None = None,
    ):
        """Initialize the research team.

        Args:
            model_name: Gemini model to use. Defaults to MODEL_NAME env var.
            searxng_url: SearXNG URL. Defaults to SEARXNG_URL env var.
        """
        self.model_name = model_name or os.getenv("MODEL_NAME", "gemini-2.5-flash")
        self.searxng_url = searxng_url or os.getenv("SEARXNG_URL", "http://localhost:8080")

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7,
        )

        # Initialize search client
        self.search_client = SearXNGClient(base_url=self.searxng_url)
        self.last_search_warning: str | None = None
        self.last_search_error: str | None = None
        self.last_rerank_warning: str | None = None
        self.last_rerank_error: str | None = None
        self.last_llm_warning: str | None = None
        self.last_llm_error: str | None = None

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.

        Returns:
            Compiled StateGraph.
        """
        # Define the graph with TeamState
        workflow = StateGraph(TeamState)

        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("writer", self._writer_node)

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Add conditional edges from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "researcher": "researcher",
                "writer": "writer",
                "finish": END,
            },
        )

        # Researcher and Writer return to Supervisor
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("writer", "supervisor")

        return workflow.compile()

    def _supervisor_node(self, state: TeamState) -> dict:
        """Supervisor agent that plans and routes tasks.

        Args:
            state: Current team state.

        Returns:
            Updated state dictionary.
        """
        # Check iteration limit
        if state.iteration >= state.max_iterations:
            return {
                "current_agent": AgentRole.SUPERVISOR,
                "iteration": state.iteration + 1,
                "error": "Max iterations reached",
            }

        # First iteration: create research plan
        if state.plan is None:
            plan = self._create_research_plan(state.query)
            return {
                "plan": plan,
                "current_agent": AgentRole.SUPERVISOR,
                "iteration": state.iteration + 1,
            }

        # Check if we have enough findings to write report
        if len(state.findings) >= len(state.plan.sub_questions):
            # Route to writer for final report
            return {
                "current_agent": AgentRole.WRITER,
                "iteration": state.iteration + 1,
            }

        # Route to researcher for next sub-question
        return {
            "current_agent": AgentRole.RESEARCHER,
            "iteration": state.iteration + 1,
        }

    def _researcher_node(self, state: TeamState) -> dict:
        """Researcher agent that executes searches and gathers information.

        Args:
            state: Current team state.

        Returns:
            Updated state dictionary.
        """
        if state.plan is None:
            return {"error": "No research plan available"}

        # Get next sub-question to research
        num_findings = len(state.findings)
        if num_findings >= len(state.plan.sub_questions):
            return {}

        sub_question = state.plan.sub_questions[num_findings]

        # Execute search
        search_results = self._execute_search(sub_question)

        # Rerank results
        reranker = Reranker()
        reranked = reranker.rerank(sub_question, search_results, top_k=5)
        self.last_rerank_warning = reranker.last_warning
        self.last_rerank_error = reranker.last_error

        # Synthesize finding from search results
        finding = self._synthesize_finding(sub_question, reranked)
        warnings = [
            warning
            for warning in (
                self.last_search_warning,
                self.last_rerank_warning,
                self.last_llm_warning,
                *(source.warning for source in reranked if source.warning),
            )
            if warning
        ]

        # Create task record
        task = ResearchTask(
            task_id=f"task-{uuid.uuid4().hex[:8]}",
            query=sub_question,
            assigned_to=AgentRole.RESEARCHER,
            status="completed",
            result=finding,
            sources=reranked,
            warnings=list(dict.fromkeys(warnings)),
            metadata={
                "search_error": self.last_search_error,
                "rerank_error": self.last_rerank_error,
                "llm_error": self.last_llm_error,
                "degraded": any(source.degraded for source in reranked) or bool(warnings),
            },
        )

        return {
            "search_results": state.search_results + reranked,
            "findings": state.findings + [finding],
            "tasks": state.tasks + [task],
            "warnings": state.warnings + list(dict.fromkeys(warnings)),
            "current_agent": AgentRole.RESEARCHER,
        }

    def _writer_node(self, state: TeamState) -> dict:
        """Writer agent that synthesizes findings into a report.

        Args:
            state: Current team state.

        Returns:
            Updated state dictionary.
        """
        if not state.findings:
            return {"error": "No findings to write about"}

        report = self._generate_report(state)

        return {
            "report": report,
            "current_agent": AgentRole.WRITER,
        }

    def _route_from_supervisor(self, state: TeamState) -> str:
        """Route from supervisor to next agent.

        Args:
            state: Current team state.

        Returns:
            Next node name.
        """
        # If we have a report, finish
        if state.report is not None:
            return "finish"

        # If there's an error, finish
        if state.error:
            return "finish"

        # Route based on current agent decision
        if state.current_agent == AgentRole.WRITER:
            return "writer"
        elif state.current_agent == AgentRole.RESEARCHER:
            return "researcher"

        return "finish"

    def _create_research_plan(self, query: str) -> ResearchPlan:
        """Create a research plan for the query.

        Args:
            query: Research query.

        Returns:
            ResearchPlan object.
        """
        system_prompt = """You are a research planning expert. Create a structured research plan.

Break down the query into 3-5 sub-questions that, when answered, will fully address the main query.
Be specific and actionable."""

        human_prompt = f"""Create a research plan for: {query}

Respond with:
1. The main objective
2. 3-5 specific sub-questions to research
3. Brief methodology
4. Expected output format"""

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )

        # Parse response into ResearchPlan
        content = response.content

        # Simple parsing (in production, use structured output)
        lines = content.split("\n")
        sub_questions = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                # Clean up the line
                clean = line.lstrip("0123456789.-•) ").strip()
                if clean and "?" in clean or len(clean) > 20:
                    sub_questions.append(clean)

        # Ensure we have at least 3 sub-questions
        if len(sub_questions) < 3:
            sub_questions = [
                f"What is {query}?",
                f"What are the key aspects of {query}?",
                f"What are the latest developments in {query}?",
            ]

        return ResearchPlan(
            objective=query,
            sub_questions=sub_questions[:5],
            methodology=["Search for information", "Analyze results", "Synthesize findings"],
            expected_output="Comprehensive research report",
        )

    def _execute_search(self, query: str) -> list[SearchResult]:
        """Execute a search query.

        Args:
            query: Search query.

        Returns:
            List of search results.
        """
        self.last_search_warning = None
        self.last_search_error = None

        # Try SearXNG first
        if self.search_client.is_available():
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            results = loop.run_until_complete(self.search_client.search(query, num_results=10))
            if results:
                return results
            self.last_search_warning = self.search_client.last_warning
            self.last_search_error = self.search_client.last_error

        # Fallback to mock results
        if self.last_search_warning is None:
            self.last_search_warning = (
                "SearXNG unavailable; using clearly marked mock search results"
            )
        logger.warning(self.last_search_warning)
        return create_mock_results(query, num_results=10)

    def _synthesize_finding(self, question: str, results: list[SearchResult]) -> str:
        """Synthesize a finding from search results.

        Args:
            question: The research question.
            results: Search results.

        Returns:
            Synthesized finding.
        """
        # Build context from results
        context_parts = []
        for i, result in enumerate(results[:5], 1):
            context_parts.append(
                f"{i}. {result.title}\n   {result.content}\n   Source: {result.url}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""Based on the following search results, answer this question: {question}

Search Results:
{context}

Provide a comprehensive answer that synthesizes information from multiple sources.
Include specific facts and cite sources where appropriate."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            self.last_llm_warning = None
            self.last_llm_error = None
            return response.content
        except Exception as exc:
            self.last_llm_error = f"Gemini finding synthesis failed: {exc}"
            self.last_llm_warning = "Gemini degraded; finding synthesized from search snippets only"
            logger.warning(self.last_llm_error)
            return (
                f"[DEGRADED: {self.last_llm_warning}]\n"
                f"Question: {question}\n\n"
                f"Available search context:\n{context}"
            )

    def _generate_report(self, state: TeamState) -> ResearchReport:
        """Generate the final research report.

        Args:
            state: Team state with findings.

        Returns:
            ResearchReport object.
        """
        if state.plan is None:
            raise ValueError("Cannot generate report without a research plan")

        # Build sections from findings
        sections = []
        all_sources = []

        for question, finding in zip(state.plan.sub_questions, state.findings, strict=False):
            # Get sources for this section
            section_sources = []
            for task in state.tasks:
                if task.result == finding:
                    section_sources = [s.url for s in task.sources]
                    all_sources.extend(task.sources)
                    break

            sections.append(
                ReportSection(
                    heading=question,
                    content=finding,
                    sources=section_sources,
                )
            )

        # Generate executive summary
        summary_prompt = f"""Write a brief executive summary (2-3 paragraphs) for a research report on: {state.query}

Key findings:
{chr(10).join(f"- {f[:200]}..." for f in state.findings)}"""

        report_warnings = list(dict.fromkeys(state.warnings))
        report_metadata: dict[str, object] = {"degraded": bool(report_warnings)}
        try:
            summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary = summary_response.content
        except Exception as exc:
            warning = "Gemini degraded; executive summary assembled from findings only"
            logger.warning("Gemini report summary failed: %s", exc)
            report_warnings = list(dict.fromkeys([*report_warnings, warning]))
            report_metadata.update({"degraded": True, "llm_error": str(exc)})
            summary = f"[DEGRADED: {warning}]\n\n" + "\n\n".join(state.findings[:3])

        return ResearchReport(
            title=f"Research Report: {state.query}",
            summary=summary,
            sections=sections,
            sources=all_sources,
            warnings=report_warnings,
            metadata=report_metadata,
        )

    async def research(self, query: str) -> ResearchReport:
        """Execute a research query.

        Args:
            query: Research query.

        Returns:
            ResearchReport with findings.
        """
        initial_state = TeamState(query=query)

        # Run the graph
        final_state = None
        async for state in self.graph.astream(initial_state):
            final_state = state

        # Extract the report from final state
        if final_state and "report" in final_state:
            return final_state["report"]

        # Handle case where graph ended without report
        if final_state:
            # Get the last value from the state dict
            for value in final_state.values():
                if isinstance(value, dict) and "report" in value:
                    return value["report"]

        raise ValueError("Research failed to produce a report")

    def research_sync(self, query: str) -> ResearchReport:
        """Synchronous version of research.

        Args:
            query: Research query.

        Returns:
            ResearchReport with findings.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.research(query))


def create_team(
    model_name: str | None = None,
    searxng_url: str | None = None,
) -> ResearchTeam:
    """Create a research team instance.

    Args:
        model_name: Gemini model name.
        searxng_url: SearXNG instance URL.

    Returns:
        ResearchTeam instance.
    """
    return ResearchTeam(model_name=model_name, searxng_url=searxng_url)
