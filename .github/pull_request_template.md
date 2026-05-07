## Summary

- 

## Verification checklist

- [ ] Ran `ruff check src/ tests/`
- [ ] Ran `ruff format --check src/ tests/`
- [ ] Ran `python -m compileall -q src tests`
- [ ] Ran relevant pytest checks, or documented why they were skipped

## Review Notes

- External services checked or intentionally skipped: SearXNG, FlashRank, Gemini/API.
- Degraded search/reranking behavior reviewed for explicit provenance, warnings, and error context.
- Gemini/API fallback or failure behavior reviewed for visible degraded labels rather than generic success.
- Skipped checks, unavailable services, or mocked dependencies are documented in the PR description.
