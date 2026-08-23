# MVP: Stress Score Dashboard

A minimal full-stack web app with a **React + Tailwind CSS frontend** and a **Python FastAPI backend**. The backend calls two JSON APIs, combines their data into a normalized **stress score from 1–100**, and returns data for a graph plus the score.

## Architecture

```text
External API #1 ──┐
                  ├──> Python FastAPI ──> React + Tailwind
External API #2 ──┘          │
                             ├── stress score (1–100)
                             └── graph-ready JSON
```

## Project structure

```text
stress-dashboard/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── package.json
│   ├── index.html
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── index.css
└── README.md
```

## Backend

### `backend/requirements.txt`

```txt
fastapi
uvicorn[standard]
httpx
python-dotenv
```

### `backend/.env`

```env
API_1_URL=https://example.com/api/one
API_2_URL=https://example.com/api/two
```

Replace the URLs with the two real JSON APIs.

### `backend/main.py`

```python
import os
import asyncio
from statistics import mean

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

API_1_URL = os.getenv("API_1_URL")
API_2_URL = os.getenv("API_2_URL")

app = FastAPI(title="Stress Score API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def fetch_json(url: str):
    if not url:
        raise RuntimeError("API URL is not configured")

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


def normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 50.0

    value = max(minimum, min(value, maximum))
    return ((value - minimum) / (maximum - minimum)) * 100


def calculate_stress(api_1: dict, api_2: dict) -> int:
    # MVP scoring example.
    # Change these fields and weights to match the real APIs.
    value_1 = float(api_1.get("value", 0))
    value_2 = float(api_2.get("value", 0))

    score_1 = normalize(value_1, 0, 100)
    score_2 = normalize(value_2, 0, 100)

    return round(mean([score_1, score_2]))


@app.get("/api/stress")
async def get_stress():
    try:
        api_1, api_2 = await asyncio.gather(
            fetch_json(API_1_URL),
            fetch_json(API_2_URL),
        )

        score = calculate_stress(api_1, api_2)

        graph = [
            {"source": "API 1", "value": float(api_1.get("value", 0))},
            {"source": "API 2", "value": float(api_2.get("value", 0))},
        ]

        return {
            "stress_score": max(1, min(100, score)),
            "graph": graph,
            "sources": {
                "api_1": api_1,
                "api_2": api_2,
            },
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to retrieve API data: {exc}",
        )
```

## Frontend

Use Vite to create the React app, then add Tailwind CSS and Recharts.

### `frontend/package.json`

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "recharts": "^2.15.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.4.0",
    "vite": "^6.0.0",
    "tailwindcss": "^4.0.0",
    "@tailwindcss/vite": "^4.0.0"
  }
}
```

### `frontend/src/App.jsx`

```jsx
import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

const API_BASE = "http://localhost:8000";

function ScoreCard({ score }) {
  const label =
    score < 34 ? "Low stress" :
    score < 67 ? "Moderate stress" :
    "High stress";

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
      <p className="text-sm font-medium uppercase tracking-wide text-slate-500">
        Stress score
      </p>

      <div className="mt-3 flex items-end gap-2">
        <span className="text-7xl font-bold tracking-tight text-slate-900">
          {score}
        </span>
        <span className="mb-2 text-xl text-slate-400">/ 100</span>
      </div>

      <p className="mt-3 text-lg font-medium text-slate-700">{label}</p>

      <div className="mt-6 h-3 overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full bg-indigo-500 transition-all duration-500"
          style={{ width: `${score}%` }}
        />
      </div>
    </div>
  );
}

function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  async function loadData() {
    try {
      setError("");

      const response = await fetch(`${API_BASE}/api/stress`);

      if (!response.ok) {
        throw new Error("Backend request failed");
      }

      const json = await response.json();
      setData(json);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  return (
    <main className="min-h-screen bg-slate-50 px-6 py-10 text-slate-900">
      <div className="mx-auto max-w-6xl">
        <header className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight">
            Stress Dashboard
          </h1>
          <p className="mt-2 text-slate-500">
            Combined data from two APIs.
          </p>
        </header>

        {error && (
          <div className="mb-6 rounded-xl border border-red-200 bg-red-50 p-4 text-red-700">
            {error}
          </div>
        )}

        {!data ? (
          <div className="rounded-2xl border border-slate-200 bg-white p-8 text-slate-500">
            Loading...
          </div>
        ) : (
          <div className="grid gap-6 lg:grid-cols-3">
            <ScoreCard score={data.stress_score} />

            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm lg:col-span-2">
              <div className="mb-5">
                <h2 className="text-lg font-semibold">API data</h2>
                <p className="text-sm text-slate-500">
                  Values returned by the two source APIs.
                </p>
              </div>

              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.graph}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="source" />
                    <YAxis />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="#6366f1"
                      strokeWidth={3}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>
        )}

        <button
          onClick={loadData}
          className="mt-6 rounded-xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700"
        >
          Refresh data
        </button>
      </div>
    </main>
  );
}

export default App;
```

### `frontend/src/main.jsx`

```jsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

### `frontend/src/index.css`

```css
@import "tailwindcss";

body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}
```

## Run locally

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Windows activation:

```powershell
.venv\Scripts\activate
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## API contract

The frontend only talks to:

```http
GET /api/stress
```

Example response:

```json
{
  "stress_score": 72,
  "graph": [
    {"source": "API 1", "value": 81},
    {"source": "API 2", "value": 63}
  ],
  "sources": {
    "api_1": {"value": 81},
    "api_2": {"value": 63}
  }
}
```

## MVP scoring logic

The first version normalizes each API's `value` to 0–100 and averages the two values:

```text
stress_score = (normalized_api_1 + normalized_api_2) / 2
```

The final score is clamped to **1–100**.

Replace `calculate_stress()` with the real scoring model and map the actual JSON fields from the two APIs.

## Recommended next steps

1. Add API keys through environment variables.
2. Validate API responses with Pydantic.
3. Add timestamps and historical graph data.
4. Store previous results in SQLite/PostgreSQL.
5. Add separate loading/error states for each API.
6. Add automatic refresh.
7. Add authentication if the data is user-specific.
8. Explain how the 1–100 score is calculated.
