"""CLI for Hierarchical Research Team.

A Typer-based command-line interface for local research workflow demos.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from research_team.agents import create_team
from research_team.models import ResearchReport
from research_team.search import SearXNGClient

app = typer.Typer(
    name="research",
    help="Local hierarchical research workflow using LangGraph, SearXNG, FlashRank, and Gemini",
    no_args_is_help=True,
)
console = Console()


def check_environment() -> bool:
    """Check if required environment variables are set.

    Returns:
        True if environment is properly configured.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] GEMINI_API_KEY environment variable not set.",
            style="bold",
        )
        console.print("Set it with: export GEMINI_API_KEY=your-api-key")
        return False
    return True


@app.command()
def research(
    query: Annotated[str, typer.Argument(help="Research query or topic to investigate")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path for the report (markdown format)"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Gemini model to use"),
    ] = "gemini-2.5-flash",
    searxng_url: Annotated[
        str,
        typer.Option("--searxng-url", "-s", help="SearXNG instance URL"),
    ] = "http://localhost:8080",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Run a local research workflow on a topic.

    This command uses a hierarchical team of AI agents to research the given query:
    - Supervisor: Plans and coordinates the research
    - Researcher: Searches and gathers information
    - Writer: Synthesizes findings into a report
    """
    if not check_environment():
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold blue]Research Query:[/bold blue] {query}",
            title="Hierarchical Research Team",
            border_style="blue",
        )
    )

    # Check SearXNG availability
    search_client = SearXNGClient(base_url=searxng_url)
    if search_client.is_available():
        console.print(f"[green]SearXNG available at {searxng_url}[/green]")
    else:
        console.print(
            f"[yellow]Warning:[/yellow] SearXNG not available at {searxng_url}. "
            "Using degraded mock results; review source metadata before trusting output."
        )

    # Create the research team
    team = create_team(model_name=model, searxng_url=searxng_url)

    # Execute research with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Researching...", total=None)

        try:
            report = asyncio.run(team.research(query))
        except Exception as e:
            console.print(f"[red]Research failed:[/red] {e}")
            raise typer.Exit(1) from e

        progress.update(task, description="Research complete!")

    # Display the report
    _display_report(report)

    # Save to file if requested
    if output:
        _save_report(report, output)
        console.print(f"\n[green]Report saved to:[/green] {output}")


def _display_report(report: ResearchReport) -> None:
    """Display the research report in the console.

    Args:
        report: The research report to display.
    """
    console.print("\n")
    console.print(
        Panel(
            f"[bold]{report.title}[/bold]",
            border_style="green",
        )
    )

    # Executive Summary
    console.print("\n[bold cyan]Executive Summary[/bold cyan]")
    console.print(Markdown(report.summary))

    # Sections
    for i, section in enumerate(report.sections, 1):
        console.print(f"\n[bold yellow]{i}. {section.heading}[/bold yellow]")
        console.print(Markdown(section.content))

        if section.sources:
            console.print("\n[dim]Sources:[/dim]")
            for source in section.sources[:3]:
                console.print(f"  [dim]- {source}[/dim]")

    # Sources summary
    if report.sources:
        console.print("\n[bold cyan]All Sources[/bold cyan]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Title", style="cyan", no_wrap=False)
        table.add_column("URL", style="blue", no_wrap=False)
        table.add_column("Score", style="green", justify="right")
        table.add_column("Provenance", style="magenta")
        table.add_column("Degraded", style="yellow")

        for source in report.sources[:10]:
            table.add_row(
                source.title[:50] + "..." if len(source.title) > 50 else source.title,
                source.url[:60] + "..." if len(source.url) > 60 else source.url,
                f"{source.score:.2f}",
                source.provenance,
                "yes" if source.degraded else "no",
            )

        console.print(table)


def _save_report(report: ResearchReport, path: Path) -> None:
    """Save the research report to a file.

    Args:
        report: The research report to save.
        path: Output file path.
    """
    content = f"# {report.title}\n\n"
    content += f"*Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    content += "## Executive Summary\n\n"
    content += report.summary + "\n\n"

    for i, section in enumerate(report.sections, 1):
        content += f"## {i}. {section.heading}\n\n"
        content += section.content + "\n\n"
        if section.sources:
            content += "**Sources:**\n"
            for source in section.sources:
                content += f"- {source}\n"
            content += "\n"

    content += "## All Sources\n\n"
    for source in report.sources:
        degraded = ", degraded" if source.degraded else ""
        content += (
            f"- [{source.title}]({source.url}) "
            f"(Score: {source.score:.2f}, provenance: {source.provenance}{degraded})\n"
        )

    path.write_text(content)


@app.command()
def status() -> None:
    """Check the status of the research team components."""
    console.print(
        Panel(
            "[bold]System Status Check[/bold]",
            border_style="blue",
        )
    )

    # Check Gemini API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        console.print("[green]Gemini API Key:[/green] Configured")
    else:
        console.print("[red]Gemini API Key:[/red] Not configured")

    # Check SearXNG
    searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080")
    search_client = SearXNGClient(base_url=searxng_url)
    if search_client.is_available():
        console.print(f"[green]SearXNG:[/green] Available at {searxng_url}")
    else:
        console.print(f"[yellow]SearXNG:[/yellow] Not available at {searxng_url}")

    # Check model
    model = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    console.print(f"[blue]Model:[/blue] {model}")


@app.command()
def demo() -> None:
    """Run a demo research query."""
    if not check_environment():
        raise typer.Exit(1)

    demo_query = (
        "What are the latest advances in large language models and their applications in 2024?"
    )

    console.print(
        Panel(
            f"[bold]Running Demo Research[/bold]\n\nQuery: {demo_query}",
            border_style="magenta",
        )
    )

    # Use mock results for demo
    team = create_team()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running demo research...", total=None)

        try:
            report = asyncio.run(team.research(demo_query))
        except Exception as e:
            console.print(f"[red]Demo failed:[/red] {e}")
            raise typer.Exit(1) from e

        progress.update(task, description="Demo complete!")

    _display_report(report)


if __name__ == "__main__":
    app()
