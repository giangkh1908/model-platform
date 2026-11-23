import { useState } from "react";
import { embed } from "../api/playground.ts";
import { Field, Spinner } from "../components/ui.tsx";
import { errorText, splitLines } from "../lib/format.ts";
import type { EmbedResponse } from "../types/api.ts";

const SAMPLE = `Tôi muốn mua laptop chơi game
Gợi ý máy tính xách tay để chơi game
Hôm nay trời mưa to quá
I want to buy a gaming laptop`;

function cellStyle(value: number, isDiagonal: boolean) {
  if (isDiagonal) return undefined;
  const strength = Math.max(0, (value - 0.3) / 0.7);
  return { background: `color-mix(in srgb, var(--ok) ${(strength * 45).toFixed(0)}%, transparent)` };
}

function shorten(text: string, length = 40): string {
  return text.length > length ? text.slice(0, length) + "…" : text;
}

export default function Embed() {
  const [text, setText] = useState(SAMPLE);
  const [result, setResult] = useState<(EmbedResponse & { lines: string[] }) | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function compute() {
    const lines = splitLines(text);
    if (!lines.length) return;
    setLoading(true);
    setError("");
    try {
      const response = await embed(lines);
      setResult({ lines, ...response });
    } catch (e) {
      setError(errorText(e));
    }
    setLoading(false);
  }

  return (
    <div className="stack">
      <Field label="Mỗi dòng một câu. Bảng cho biết các câu giống nhau đến đâu (1 = giống hệt).">
        <textarea rows={5} value={text} onChange={(e) => setText(e.target.value)} />
      </Field>
      <div className="row wrap">
        <button type="button" className="btn primary" onClick={compute} disabled={loading}>
          {loading && <Spinner />} Tính embedding
        </button>
        {result && (
          <span className="muted small">
            {result._ms} ms · vector {result.dims} chiều · {result.usage?.prompt_tokens ?? "?"} token
          </span>
        )}
        {error && <span className="error-text small">{error}</span>}
      </div>
      {result && (
        <>
          <div className="scroll-x">
            <table className="sim">
              <thead>
                <tr>
                  <th />
                  {result.lines.map((_, i) => (
                    <th key={i}>{i + 1}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.similarity.map((row, i) => (
                  <tr key={i}>
                    <th title={result.lines[i]} className="left">
                      {i + 1}. {shorten(result.lines[i])}
                    </th>
                    {row.map((value, j) => (
                      <td key={j} style={cellStyle(value, i === j)}>
                        {value.toFixed(3)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details>
            <summary className="muted small">8 giá trị đầu của mỗi vector</summary>
            <pre className="code">
              {result.preview.map((v, i) => `${i + 1}. [${v.join(", ")}, …]`).join("\n")}
            </pre>
          </details>
        </>
      )}
    </div>
  );
}
