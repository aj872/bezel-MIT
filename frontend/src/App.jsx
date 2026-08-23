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