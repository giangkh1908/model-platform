import { useState } from "react";
import { Download } from "lucide-react";
import { generateImage } from "../api/playground.ts";
import { Field, Spinner } from "../components/ui.tsx";
import { errorText } from "../lib/format.ts";
import type { ImageResponse } from "../types/api.ts";

const SIZES = ["512x512", "768x768", "1024x1024", "1024x768", "768x1024"];
const SAMPLE = "a cozy cafe in Hanoi at sunset, watercolor style";

function optionalInt(value: string): number | undefined {
  if (value === "") return undefined;
  return Number(value);
}

function ResultImage({ base64, ms }: { base64: string; ms: number }) {
  const src = `data:image/png;base64,${base64}`;
  return (
    <div className="stack">
      <div className="row wrap">
        <span className="muted small">{Math.round(ms / 100) / 10} s</span>
        <a className="btn sm" href={src} download="anh-sinh.png">
          <Download size={14} /> Tải ảnh
        </a>
      </div>
      <img src={src} alt="Ảnh vừa sinh" className="generated-image" />
    </div>
  );
}

export default function ImageGen() {
  const [prompt, setPrompt] = useState(SAMPLE);
  const [size, setSize] = useState("1024x1024");
  const [steps, setSteps] = useState("");
  const [seed, setSeed] = useState("");
  const [result, setResult] = useState<ImageResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function generate() {
    if (!prompt.trim()) return;
    setLoading(true);
    setError("");
    try {
      setResult(await generateImage({ prompt: prompt.trim(), size, steps: optionalInt(steps), seed: optionalInt(seed) }));
    } catch (e) {
      setError(errorText(e));
    }
    setLoading(false);
  }

  return (
    <div className="stack">
      <Field label="Mô tả ảnh (prompt)" hint="Model SDXL hiểu tiếng Anh tốt hơn tiếng Việt.">
        <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
      </Field>
      <div className="row wrap">
        <Field label="Kích thước">
          <select value={size} onChange={(e) => setSize(e.target.value)}>
            {SIZES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Số bước (để trống = mặc định)">
          <input type="number" min="1" max="100" value={steps} onChange={(e) => setSteps(e.target.value)} />
        </Field>
        <Field label="Seed (để trống = ngẫu nhiên)">
          <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
        </Field>
      </div>
      <div className="row wrap">
        <button type="button" className="btn primary" onClick={generate} disabled={loading || !prompt.trim()}>
          {loading && <Spinner />} Sinh ảnh
        </button>
        {loading && <span className="muted small">Đang sinh ảnh, có thể mất vài chục giây…</span>}
        {error && <span className="error-text small">{error}</span>}
      </div>
      {result && <ResultImage base64={result.image} ms={result._ms} />}
    </div>
  );
}
