# Hierarchical Research Team

A production-ready AI-powered research automation system using multi-agent orchestration. The system employs a hierarchical team of specialized agents that collaborate to conduct comprehensive research on any topic.

## Features

- **Multi-Agent Architecture**: Supervisor, Researcher, and Writer agents with distinct roles
- **Self-Hosted Search**: SearXNG integration for privacy-focused meta-search (no per-query costs)
- **Smart Reranking**: FlashRank for CPU-efficient document relevance scoring
- **Structured Output**: Generates well-organized research reports with citations
- **Async Support**: Built with async/await for high performance
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

Self-hosted meta-search engine that aggregates results from multiple sources:

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
