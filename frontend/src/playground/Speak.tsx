import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { Download, Mic, Square, Upload } from "lucide-react";
import { speak as speakApi, ttsVoices } from "../api/playground.ts";
import { Field, Spinner } from "../components/ui.tsx";
import { encodeWav } from "../lib/browser.ts";
import { errorText } from "../lib/format.ts";
import type { Voice } from "../types/api.ts";

/** Giọng mẫu để clone. */
interface Clip {
  name: string;
  blob: Blob;
}

const SAMPLE = "Xin chào, hôm nay trời đẹp quá. Mình đang thử giọng nói tiếng Việt của mô hình này.";
const MAX_RECORD_SECONDS = 8;

function readAsBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function base64ToBlobUrl(base64: string): string {
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
}

// Bản ghi của trình duyệt (webm/ogg) được giải mã rồi đóng lại thành WAV để gửi lên máy chủ
async function blobToWav(blob: Blob): Promise<Blob> {
  const context = new AudioContext();
  const decoded = await context.decodeAudioData(await blob.arrayBuffer());
  await context.close();
  return encodeWav(decoded.getChannelData(0), decoded.sampleRate);
}

export default function Speak() {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [voice, setVoice] = useState("");
  const [clip, setClip] = useState<Clip | null>(null);
  const [clipUrl, setClipUrl] = useState("");
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [text, setText] = useState(SAMPLE);
  const [result, setResult] = useState<{ url: string; ms: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const tick = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  const autoStop = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    ttsVoices()
      .then((list) => {
        setVoices(list);
        if (list.length) setVoice(list[0].id);
      })
      .catch(() => setVoices([]));
  }, []);

  useEffect(() => {
    if (!clip) {
      setClipUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(clip.blob);
    setClipUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [clip]);

  useEffect(() => {
    return () => {
      if (result) URL.revokeObjectURL(result.url);
    };
  }, [result]);

  useEffect(() => {
    return () => {
      clearInterval(tick.current);
      clearTimeout(autoStop.current);
      if (recorder.current?.state === "recording") recorder.current.stop();
    };
  }, []);

  function stopRecording() {
    clearTimeout(autoStop.current);
    clearInterval(tick.current);
    if (recorder.current?.state === "recording") recorder.current.stop();
  }

  async function startRecording() {
    setError("");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("Không mở được micro. Kiểm tra quyền micro của trình duyệt.");
      return;
    }
    chunks.current = [];
    const rec = new MediaRecorder(stream);
    rec.ondataavailable = (e) => {
      if (e.data.size) chunks.current.push(e.data);
    };
    rec.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      setRecording(false);
      try {
        const wav = await blobToWav(new Blob(chunks.current, { type: rec.mimeType }));
        setClip({ name: "giong-thu.wav", blob: wav });
      } catch {
        setError("Không đọc được bản ghi. Thử thu lại.");
      }
    };
    recorder.current = rec;
    rec.start();
    setSeconds(0);
    setRecording(true);
    tick.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    autoStop.current = setTimeout(stopRecording, MAX_RECORD_SECONDS * 1000);
  }

  function pickFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) setClip({ name: file.name, blob: file });
    event.target.value = "";
  }

  async function speak() {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    try {
      const body: Record<string, string> = { input: text.trim() };
      if (clip) body.ref_audio = await readAsBase64(clip.blob);
      else body.voice = voice;
      const response = await speakApi(body);
      setResult({ url: base64ToBlobUrl(response.audio), ms: response._ms });
    } catch (e) {
      setError(errorText(e));
    }
    setLoading(false);
  }

  return (
    <div className="stack">
      <Field label="Nội dung cần đọc" hint="Tối đa 1000 ký tự.">
        <textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} />
      </Field>

      <Field label="Clone giọng" hint="Thu 3–8 giây, một người nói, không có tiếng ồn. Có giọng mẫu thì giọng dựng sẵn bị bỏ qua.">
        <div className="row wrap">
          {recording ? (
            <button type="button" className="btn danger" onClick={stopRecording}>
              <Square size={14} /> Dừng ({seconds}s / {MAX_RECORD_SECONDS}s)
            </button>
          ) : (
            <button type="button" className="btn" onClick={startRecording}>
              <Mic size={14} /> Thu giọng
            </button>
          )}
          <label className="btn">
            <Upload size={14} /> Chọn file WAV
            <input type="file" accept=".wav,audio/wav" hidden onChange={pickFile} />
          </label>
          {clip && (
            <button type="button" className="btn ghost sm" onClick={() => setClip(null)}>
              Bỏ giọng mẫu
            </button>
          )}
        </div>
        {clip && (
          <div className="row wrap">
            <span className="small muted">Giọng mẫu: {clip.name}</span>
            <audio controls src={clipUrl} />
          </div>
        )}
      </Field>

      <Field label="Giọng dựng sẵn" hint={clip ? "Đang dùng giọng mẫu ở trên." : ""}>
        <select value={voice} onChange={(e) => setVoice(e.target.value)} disabled={!!clip || voices.length === 0}>
          {voices.map((v) => (
            <option key={v.id} value={v.id}>
              {v.label}
            </option>
          ))}
        </select>
      </Field>

      <div className="row wrap">
        <button type="button" className="btn primary" onClick={speak} disabled={loading || recording || !text.trim()}>
          {loading && <Spinner />} Đọc
        </button>
        {loading && <span className="muted small">Đang tạo giọng nói…</span>}
        {error && <span className="error-text small">{error}</span>}
      </div>
      {result && (
        <div className="stack">
          <div className="row wrap">
            <span className="muted small">{Math.round(result.ms / 100) / 10} s</span>
            <a className="btn sm" href={result.url} download="giong-noi.wav">
              <Download size={14} /> Tải file
            </a>
          </div>
          <audio controls src={result.url} />
        </div>
      )}
    </div>
  );
}
