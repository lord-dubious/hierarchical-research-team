# Hierarchical Research Team

A small portfolio project that experiments with LangGraph-based research workflows, SearXNG search, FlashRank reranking, and Gemini-generated summaries. It is useful for local demos and code review, but external service failures can degrade output quality and should be treated as visible warnings rather than successful research.

## What Works Today

- Creates a simple research plan, executes sub-question searches, reranks returned snippets, and assembles a markdown-style report.
- Uses Pydantic models to carry source, warning, error, and degraded-path metadata through search results, tasks, and reports.
- Provides a Typer CLI plus a Python API for local experiments.
- Includes tests for model behavior, SearXNG error handling, FlashRank fallback paths, and agent workflow basics.

## Current Limits

- This is not a production research system and does not verify facts beyond the snippets returned by search providers.
- Mock search results are used when SearXNG is unavailable, so reports can be illustrative rather than evidence-backed.
- Gemini calls can fail because of missing credentials, network issues, quota limits, model changes, or API behavior.
- The report writer uses simple prompting and does not guarantee complete citations, source quality, or benchmarked accuracy.

## Features

- **Multi-Agent Architecture**: Supervisor, Researcher, and Writer agents with distinct roles
- **Self-Hosted Search**: SearXNG integration for privacy-focused meta-search (no per-query costs)
- **Smart Reranking**: FlashRank for CPU-efficient document relevance scoring
- **Structured Output**: Generates report-shaped output with source metadata
- **Async Support**: Uses async/await around the graph and search client
- **Observable**: Phoenix tracing integration for monitoring agent behavior

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Supervisor                            │
│  - Creates research plan                                     │
│  - Routes tasks to specialized agents                        │
│  - Monitors progress and decides completion                  │
└─────────────────┬───────────────────────┬───────────────────┘
                  │                       │
                  ▼                       ▼
    ┌─────────────────────┐   ┌─────────────────────┐
    │     Researcher      │   │       Writer        │
    │  - Executes search  │   │  - Synthesizes      │
    │  - Reranks results  │   │    findings         │
    │  - Extracts facts   │   │  - Generates report │
    └─────────────────────┘   └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  SearXNG + FlashRank │
    │  (Search & Rerank)   │
    └─────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for SearXNG)
- Gemini API key

### Installation

```bash
# Clone the repository
git clone https://github.com/lord-dubious/hierarchical-research-team.git
cd hierarchical-research-team

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Dependency Behavior

- **SearXNG**: Expected at `SEARXNG_URL`. If it is down or returns an HTTP error, the client records `last_error`, `last_warning`, and `last_degraded`; agent flows fall back to clearly marked mock results.
- **FlashRank**: Imported lazily. If it is missing or reranking fails, results are scored with keyword-overlap fallback metadata and warnings.
- **Gemini/API**: Requires `GEMINI_API_KEY`. Finding/report fallbacks include degraded labels and warning metadata when model calls fail inside the workflow.
- **Phoenix tracing**: Listed as a dependency for observability experiments; it is not required to validate output quality.

## Search/Reranking Boundaries

- Search result `score` values are not comparable across live SearXNG, mock data, FlashRank, and fallback reranking paths.
- Results with `degraded=True`, `provenance="mock"`, or `provenance="fallback"` should not be interpreted as high-confidence evidence.
- Empty live search results can mean SearXNG returned no matches or that a service failure was recorded on the client state.
- Reranking only reorders snippets already returned by search; it does not validate source accuracy or completeness.

### Start SearXNG (Optional but Recommended)

```bash
# Start SearXNG with docker-compose
docker-compose up -d

# Verify it's running
curl http://localhost:8080/healthz
```

### Run Research

```bash
# Using CLI
research "What are the latest advances in quantum computing?"

# With output file
research "Impact of AI on healthcare" --output report.md

# Check system status
research status
```

### Python API

```python
import asyncio
from research_team import create_team

async def main():
    team = create_team()
    report = await team.research(
        "What are the environmental impacts of electric vehicles?"
    )
    print(f"Title: {report.title}")
    print(f"Summary: {report.summary}")
    for section in report.sections:
        print(f"\n## {section.heading}")
        print(section.content)

asyncio.run(main())
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google AI API key (required) | - |
| `MODEL_NAME` | Gemini model to use | `gemini-2.5-flash` |
| `SEARXNG_URL` | SearXNG instance URL | `http://localhost:8080` |
| `SEARXNG_TIMEOUT` | Search timeout in seconds | `10` |

### CLI Options

```bash
research --help

Usage: research [OPTIONS] QUERY

Options:
  -o, --output PATH      Output file for report (markdown)
  -m, --model TEXT       Gemini model name
  -s, --searxng-url TEXT SearXNG instance URL
  -v, --verbose          Enable verbose output
  --help                 Show this message and exit
```

## Components

### SearXNG Client

Self-hosted meta-search client. Check result metadata and client state when results are empty:

```python
from research_team import SearXNGClient

client = SearXNGClient(base_url="http://localhost:8080")
results = await client.search("machine learning", num_results=10)
```

### FlashRank Reranker

CPU-efficient document reranking (4MB model):

```python
from research_team import Reranker, rerank_results

reranker = Reranker()
ranked = reranker.rerank("AI ethics", search_results, top_k=5)

# Or use convenience function
ranked = rerank_results("AI ethics", search_results, top_k=5)
```

Fallback reranking sets `degraded=True`, `provenance="fallback"`, and warning metadata so downstream code can distinguish it from FlashRank scoring.

### Research Team

Multi-agent workflow with LangGraph:

```python
from research_team import ResearchTeam

team = ResearchTeam(
    model_name="gemini-2.5-flash",
    searxng_url="http://localhost:8080"
)

# Async
report = await team.research("Topic")

# Sync
report = team.research_sync("Topic")
```

## Development

### Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=research_team

# Run specific test file
pytest tests/test_agents.py -v
```

### Linting

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

## Project Structure

```
hierarchical-research-team/
├── src/research_team/
│   ├── __init__.py      # Package exports
│   ├── models.py        # Pydantic models
│   ├── search.py        # SearXNG client
│   ├── reranker.py      # FlashRank wrapper
│   ├── agents.py        # LangGraph workflow
│   └── cli.py           # Typer CLI
├── tests/
│   ├── conftest.py      # Test fixtures
│   ├── test_models.py   # Model tests
│   ├── test_search.py   # Search client tests
│   ├── test_reranker.py # Reranker tests
│   └── test_agents.py   # Agent workflow tests
├── docker-compose.yml   # SearXNG setup
├── Dockerfile           # Container build
├── pyproject.toml       # Project config
└── README.md
```

## Tech Stack

- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - Multi-agent orchestration
- **[SearXNG](https://docs.searxng.org/)** - Self-hosted meta-search
- **[FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)** - CPU-based reranking
- **[Gemini](https://ai.google.dev/)** - AI reasoning (2.5 Flash)
- **[Typer](https://typer.tiangolo.com/)** - CLI framework
- **[Rich](https://rich.readthedocs.io/)** - Terminal formatting
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Built by [lord-dubious](https://github.com/lord-dubious)
