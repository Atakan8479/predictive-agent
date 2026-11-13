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

## Local infrastructure

Docker Desktop must be running with WSL2 integration enabled. Start the M1 infrastructure stack
and wait up to 60 seconds for every service to become healthy:

```bash
make up
make ps
```

The default host endpoints are:

| Service | Endpoint | Purpose |
|---|---|---|
| Valkey | `redis://localhost:6379` | Persistent cache |
| NATS | `nats://localhost:4222` | JetStream event bus |
| NATS monitoring | `http://localhost:8222` | Health and JetStream diagnostics |
| Toxiproxy API | `http://localhost:8474` | Proxy management |
| Proxied mock APIs | `http://localhost:18100` | Mock APIs with the active fault profile |
| Jaeger OTLP | `localhost:4317` (gRPC), `http://localhost:4318` (HTTP) | Trace ingestion |
| Jaeger UI | `http://localhost:16686` | Trace inspection |
| Mock APIs | `http://localhost:8080` | Deterministic OMS, shipping, CRM, policy, and inventory data |

Host binding and ports can be changed through the `PA_INFRA_*`, `PA_VALKEY_PORT`, `PA_NATS_*`,
`PA_TOXIPROXY_*`, `PA_JAEGER_*`, and `PA_MOCK_APIS_PORT` values in `.env`. Valkey and NATS data use
named volumes and survive `make down`. Use `make logs` to follow service logs and `make down` to
stop the stack.

Apply a response-direction fault profile to the proxied mock APIs:

```bash
make profile P=slow
uv run pa-toxiproxy inspect
uv run pa-toxiproxy reset
```

Available profiles are `fast` (50±10 ms), `typical` (200±50 ms), `slow` (800±200 ms), and
`flaky` (typical latency plus 5% connection resets and 2% timeouts). Latency is applied only to
the downstream response stream, so a profile's latency is its target contribution to request RTT.

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
