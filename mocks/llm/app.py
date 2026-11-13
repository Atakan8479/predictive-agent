"""FastAPI application for the deterministic mock LLM."""

from fastapi import FastAPI

app = FastAPI(title="Predictive Agent Mock LLM", version="1.0.0")


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}
