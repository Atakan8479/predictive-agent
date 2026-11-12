# Predictive Agent

An event-driven, asynchronous LLM agent orchestration platform. A fast router predicts user
intent and fetches required external data in the background **before** the main agent starts,
placing it in an in-memory cache. The main agent reads this data without waiting for network
latency (predictive tool prefetching).

> **Status:** M0 is complete; M1 is in progress. See [ROADMAP.md](ROADMAP.md) for the roadmap
> and [docs/adr/](docs/adr/) for architecture decisions.

## Requirements

- WSL2 (Ubuntu 24.04), `networkingMode=mirrored`
- [uv](https://docs.astral.sh/uv/), Docker, and [Ollama](https://ollama.com) running on Windows

## Setup

```bash
cp .env.example .env
uv sync
uv run pre-commit install
```

## Environment check and LLM baseline

```bash
uv run pa-doctor --runs 5 --json doctor-uncached.json
uv run pa-doctor --runs 5 --cached --json doctor-cached.json
```

The doctor checks Ollama connectivity and model availability, warms the model with `keep_alive`,
and measures warm generation performance. By default, every measured prompt begins with a unique
UUID to measure uncached prefill. With `--cached`, one unmeasured request primes the exact benchmark
prompt before the requested number of identical measured runs. The CLI and JSON reports label each
mode explicitly and summarize the median, minimum, and maximum.

## Development

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pyright           # type checking
```

Every change follows: feature branch → PR → green CI → merge. Commit messages use
[Conventional Commits](https://www.conventionalcommits.org/).
