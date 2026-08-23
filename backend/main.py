"""FastAPI service that combines two upstream JSON measurements into a stress score."""

import asyncio
import os
from contextlib import asynccontextmanager
from statistics import mean
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()


def setting(name: str, default: str | None = None) -> str | None:
    """Read a setting at call time, which also makes configuration easy to test."""
    return os.getenv(name, default)


def get_path(payload: Any, path: str) -> Any:
    """Retrieve a nested value using a dot-separated path, e.g. ``data.value``."""
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Field '{path}' was not present in the upstream response")
        value = value[part]
    return value


def normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        raise ValueError("Maximum must be greater than minimum")
    clamped = max(minimum, min(value, maximum))
    return (clamped - minimum) / (maximum - minimum) * 100


def calculate_stress(api_1: Any, api_2: Any) -> tuple[int, list[dict[str, float | str]]]:
    """Return a bounded 1–100 score and graph-ready source measurements.

    Configure each source's JSON field and expected range in ``.env``.  The
    defaults expect the simple ``{\"value\": number}`` shape used by the MVP.
    """
    configs = (
        ("API_1", "API 1", api_1),
        ("API_2", "API 2", api_2),
    )
    graph: list[dict[str, float | str]] = []
    normalized: list[float] = []

    for prefix, label, payload in configs:
        field = setting(f"{prefix}_VALUE_PATH", "value") or "value"
        try:
            raw_value = float(get_path(payload, field))
            minimum = float(setting(f"{prefix}_MIN", "0") or 0)
            maximum = float(setting(f"{prefix}_MAX", "100") or 100)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} returned an invalid measurement: {exc}") from exc

        normalized.append(normalize(raw_value, minimum, maximum))
        graph.append({"source": label, "value": raw_value, "normalized_value": round(normalized[-1], 2)})

    return max(1, min(100, round(mean(normalized)))), graph


async def fetch_json(client: httpx.AsyncClient, url: str | None) -> Any:
    if not url:
        raise ValueError("An upstream API URL is not configured")
    response = await client.get(url)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise ValueError("An upstream API returned invalid JSON") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="Stress Score API", version="0.1.0", lifespan=lifespan)

allowed_origins = [origin.strip() for origin in (setting("CORS_ORIGINS", "http://localhost:5173") or "").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stress")
async def get_stress() -> dict[str, Any]:
    try:
        api_1, api_2 = await asyncio.gather(
            fetch_json(app.state.http_client, setting("API_1_URL")),
            fetch_json(app.state.http_client, setting("API_2_URL")),
        )
        score, graph = calculate_stress(api_1, api_2)
        return {"stress_score": score, "graph": graph, "sources": {"api_1": api_1, "api_2": api_2}}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Unable to retrieve upstream API data") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

