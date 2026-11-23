import { useState } from "react";
import { rerank } from "../api/playground.ts";
import { Field, ScoreBars, Spinner } from "../components/ui.tsx";
import { errorText, splitLines } from "../lib/format.ts";

const SAMPLE_DOCS = `Vệ sinh quạt và thay keo tản nhiệt giúp card đồ họa mát hơn.
Giới hạn điện năng GPU bằng MSI Afterburner làm giảm nhiệt.
Công thức nấu phở bò truyền thống.`;

export default function Rerank() {
  const [query, setQuery] = useState("Cách giảm nhiệt độ GPU");
  const [docsText, setDocsText] = useState(SAMPLE_DOCS);
  const [result, setResult] = useState<{ ms: number; ranked: [string, number][] } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function rank() {
    const documents = splitLines(docsText);
    if (!documents.length) return;
    setLoading(true);
    setError("");
    try {
      const response = await rerank(query, documents);
      const ranked = (response.results ?? [])
        .slice()
        .sort((a, b) => b.relevance_score - a.relevance_score)
        .map((r): [string, number] => [documents[r.index], r.relevance_score]);
      setResult({ ms: response._ms, ranked });
    } catch (e) {
      setError(errorText(e));
    }
    setLoading(false);
  }

  const maxScore = result ? Math.max(...result.ranked.map((r) => r[1]), 1e-9) : 1;

  return (
    <div className="stack">
      <Field label="Câu truy vấn">
        <input value={query} onChange={(e) => setQuery(e.target.value)} />
      </Field>
      <Field label="Tài liệu, mỗi dòng một tài liệu">
        <textarea rows={5} value={docsText} onChange={(e) => setDocsText(e.target.value)} />
      </Field>
      <div className="row wrap">
        <button type="button" className="btn primary" onClick={rank} disabled={loading}>
          {loading && <Spinner />} Xếp hạng
        </button>
        {result && <span className="muted small">{result.ms} ms</span>}
        {error && <span className="error-text small">{error}</span>}
      </div>
      {result && <ScoreBars rows={result.ranked} top={result.ranked[0]?.[0]} max={maxScore} format={(v) => v.toFixed(3)} />}
    </div>
  );
}
