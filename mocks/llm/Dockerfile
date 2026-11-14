FROM ghcr.io/astral-sh/uv:0.13.6 AS uv
FROM python:3.12.16-slim

WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable
COPY mocks ./mocks

USER 65534:65534
EXPOSE 8200
CMD ["/app/.venv/bin/uvicorn", "mocks.llm.app:app", "--host", "0.0.0.0", "--port", "8200"]
