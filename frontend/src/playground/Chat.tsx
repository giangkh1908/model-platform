import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Brain, RotateCcw, Send, Settings2, Square, Trash2 } from "lucide-react";
import { chatStream } from "../api/playground.ts";
import { CopyButton, Field, Segmented, Spinner } from "../components/ui.tsx";
import { loadLocal, saveLocal } from "../lib/browser.ts";
import type { PanelProps } from "../pages/Playground.tsx";

type ThinkingMode = "auto" | "on" | "off";

interface ChatParams {
  system: string;
  temperature: number | string;
  top_p: number | string;
  max_tokens: number | string;
  thinking: ThinkingMode;
}

interface ChatStats {
  ms: number;
  ttft: number | null;
  tokens: number;
  tps: number | null;
  stopped: boolean;
}

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  reasoning?: string;
  stats?: ChatStats;
  finish?: string;
  error?: string;
}

/** Một sự kiện SSE kiểu OpenAI (chỉ các trường giao diện đọc). */
interface StreamChunk {
  error?: unknown;
  usage?: { completion_tokens?: number };
  choices?: { finish_reason?: string | null; delta?: { content?: string; reasoning_content?: string; reasoning?: string } }[];
}

const DEFAULT_PARAMS: ChatParams = { system: "", temperature: 0.7, top_p: 0.95, max_tokens: 4096, thinking: "auto" };

const THINKING_OPTIONS: { id: ThinkingMode; label: string }[] = [
  { id: "auto", label: "Mặc định" },
  { id: "on", label: "Bật" },
  { id: "off", label: "Tắt" },
];

// Gửi lên backend: null = để model tự quyết, true/false = bật/tắt qua chat template.
function thinkingFlag(value: ThinkingMode): boolean | null {
  if (value === "on") return true;
  if (value === "off") return false;
  return null;
}

// Một số model trả phần suy nghĩ ngay trong nội dung, bọc bởi <think>…</think>.
function splitThinking(content: string): { reasoning: string; answer: string } {
  const match = content.match(/^\s*<think>([\s\S]*?)(<\/think>|$)/);
  if (!match) return { reasoning: "", answer: content };
  return { reasoning: match[1].trim(), answer: content.slice(match[0].length).trimStart() };
}

async function* readSse(response: Response): AsyncGenerator<StreamChunk> {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() ?? "";
    for (const event of events) {
      for (const line of event.split(/\r?\n/)) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        try {
          yield JSON.parse(data) as StreamChunk;
        } catch {
          // dòng không phải JSON: bỏ qua
        }
      }
    }
  }
}

function statsText(stats: ChatStats | undefined): string {
  if (!stats) return "";
  const parts = [`${(stats.ms / 1000).toFixed(1)} s`];
  if (stats.ttft != null) parts.push(`chữ đầu sau ${stats.ttft} ms`);
  if (stats.tokens) parts.push(`${stats.tokens} token`);
  if (stats.tps) parts.push(`${stats.tps} token/giây`);
  if (stats.stopped) parts.push("đã dừng");
  return parts.join(" · ");
}

function truncationWarning(message: ChatMessage, answer: string): string {
  if (message.finish !== "length") return "";
  if (!answer) {
    return "Model hết giới hạn token khi vẫn đang suy nghĩ nên chưa kịp trả lời. Tăng \"Tối đa token\" trong Tham số, hoặc tắt Thinking.";
  }
  return "Câu trả lời bị cắt vì chạm giới hạn \"Tối đa token\".";
}

function Message({ message, live }: { message: ChatMessage; live: boolean }) {
  const [showThinking, setShowThinking] = useState(false);
  const { reasoning: inlineReasoning, answer } = splitThinking(message.content ?? "");
  const reasoning = message.reasoning || inlineReasoning;
  const thinkingNow = live && reasoning && !answer;
  const warning = truncationWarning(message, answer);

  if (message.role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{message.content}</div>
      </div>
    );
  }

  let body: ReactNode = <span className="muted">…</span>;
  if (answer) body = <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>;
  else if (thinkingNow) body = null;
  else if (!live && !message.error) body = <span className="muted">(không có câu trả lời)</span>;

  return (
    <div className="msg assistant">
      {reasoning && (
        <div className="thinking">
          <button type="button" className="btn ghost sm" onClick={() => setShowThinking(!showThinking)}>
            {thinkingNow ? <Spinner /> : <Brain size={14} />}
            {thinkingNow ? "Đang suy nghĩ…" : "Suy nghĩ"}
            <span className="muted">· {showThinking ? "ẩn" : "xem"}</span>
          </button>
          {showThinking && <div className="thinking-text">{reasoning}</div>}
        </div>
      )}
      {body && <div className="markdown">{body}</div>}
      {warning && <div className="warn-text small">{warning}</div>}
      {message.error && <div className="error-text small">{message.error}</div>}
      <div className="msg-foot">
        <span className="muted small">{statsText(message.stats)}</span>
        {answer && <CopyButton text={answer} label="" />}
      </div>
    </div>
  );
}

export default function Chat({ modelId }: PanelProps) {
  const storageKey = `chat:${modelId}`;
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadLocal<ChatMessage[]>(storageKey, []));
  const [params, setParams] = useState<ChatParams>(() => ({ ...DEFAULT_PARAMS, ...loadLocal<Partial<ChatParams>>("chat:params", {}) }));
  const [input, setInput] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  // Chỉ tự cuộn xuống khi người dùng đang ở cuối khung chat, để còn kéo lên đọc được.
  const stickToBottom = useRef(true);

  useEffect(() => {
    if (!streaming) saveLocal(storageKey, messages);
  }, [messages, streaming, storageKey]);

  useEffect(() => saveLocal("chat:params", params), [params]);

  useEffect(() => {
    const log = logRef.current;
    if (log && stickToBottom.current) log.scrollTop = log.scrollHeight;
  }, [messages]);

  useEffect(() => () => abortRef.current?.abort(), []);

  function onLogScroll() {
    const log = logRef.current;
    if (!log) return;
    stickToBottom.current = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  }

  function updateLast(patch: Partial<ChatMessage> | ((last: ChatMessage) => Partial<ChatMessage>)) {
    setMessages((list) => {
      const copy = list.slice();
      const last = copy[copy.length - 1];
      copy[copy.length - 1] = { ...last, ...(typeof patch === "function" ? patch(last) : patch) };
      return copy;
    });
  }

  async function generate(history: ChatMessage[]) {
    const payload = history.map(({ role, content }) => ({ role, content }));
    if (params.system.trim()) payload.unshift({ role: "system", content: params.system.trim() });

    stickToBottom.current = true;
    setMessages([...history, { role: "assistant", content: "", reasoning: "" }]);
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    const started = performance.now();
    let firstTokenAt: number | null = null;
    let completionTokens = 0;

    function finish(stopped: boolean) {
      const genSeconds = firstTokenAt ? (performance.now() - firstTokenAt) / 1000 : 0;
      updateLast({
        stats: {
          ms: performance.now() - started,
          ttft: firstTokenAt ? Math.round(firstTokenAt - started) : null,
          tokens: completionTokens,
          tps: completionTokens && genSeconds ? Number((completionTokens / genSeconds).toFixed(1)) : null,
          stopped,
        },
      });
    }

    try {
      const response = await chatStream(
        {
          messages: payload,
          temperature: Number(params.temperature),
          top_p: Number(params.top_p),
          max_tokens: Number(params.max_tokens),
          thinking: thinkingFlag(params.thinking),
        },
        controller.signal,
      );
      for await (const chunk of readSse(response)) {
        if (chunk.error) throw new Error(typeof chunk.error === "string" ? chunk.error : JSON.stringify(chunk.error));
        if (chunk.usage?.completion_tokens) completionTokens = chunk.usage.completion_tokens;
        const choice = chunk.choices?.[0];
        const finishReason = choice?.finish_reason;
        if (finishReason) updateLast({ finish: finishReason });
        const delta = choice?.delta;
        if (!delta) continue;
        const text = delta.content ?? "";
        const thought = delta.reasoning_content ?? delta.reasoning ?? "";
        if (!text && !thought) continue;
        if (firstTokenAt == null) firstTokenAt = performance.now();
        updateLast((last) => ({ content: last.content + text, reasoning: (last.reasoning ?? "") + thought }));
      }
      finish(false);
    } catch (e) {
      if (e instanceof Error && e.name === "AbortError") finish(true);
      else updateLast({ error: e instanceof Error ? e.message : String(e) });
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function send() {
    const text = input.trim();
    if (!text || streaming) return;
    setInput("");
    generate([...messages, { role: "user", content: text }]);
  }

  function regenerate() {
    const lastUser = messages.map((m) => m.role).lastIndexOf("user");
    if (lastUser < 0 || streaming) return;
    generate(messages.slice(0, lastUser + 1));
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function setParam<K extends keyof ChatParams>(key: K, value: ChatParams[K]) {
    setParams((p) => ({ ...p, [key]: value }));
  }

  return (
    <div className="chat">
      <div className="row between wrap">
        <div className="row wrap" title="Bật/tắt chế độ suy nghĩ của model (Qwen3, DeepSeek-R1…). Mặc định: theo model.">
          <Brain size={16} />
          <span className="small strong">Thinking</span>
          <Segmented value={params.thinking} options={THINKING_OPTIONS} onChange={(v) => setParam("thinking", v)} />
        </div>
        <div className="row wrap">
          <button type="button" className="btn ghost sm" onClick={() => setShowSettings(!showSettings)}>
            <Settings2 size={14} /> Tham số
          </button>
          <button type="button" className="btn ghost sm" onClick={regenerate} disabled={streaming || !messages.length}>
            <RotateCcw size={14} /> Tạo lại
          </button>
          <button type="button" className="btn ghost sm" onClick={() => setMessages([])} disabled={streaming || !messages.length}>
            <Trash2 size={14} /> Xoá hội thoại
          </button>
        </div>
      </div>

      {showSettings && (
        <div className="chat-settings">
          <Field label="System prompt">
            <textarea rows={2} value={params.system} onChange={(e) => setParam("system", e.target.value)} placeholder="Ví dụ: Bạn là trợ lý trả lời ngắn gọn bằng tiếng Việt." />
          </Field>
          <div className="form-grid three">
            <Field label={`Temperature: ${params.temperature}`}>
              <input type="range" min="0" max="2" step="0.05" value={params.temperature} onChange={(e) => setParam("temperature", e.target.value)} />
            </Field>
            <Field label={`Top P: ${params.top_p}`}>
              <input type="range" min="0.05" max="1" step="0.05" value={params.top_p} onChange={(e) => setParam("top_p", e.target.value)} />
            </Field>
            <Field label="Tối đa token" hint="Gồm cả phần suy nghĩ. Model thinking cần nhiều hơn.">
              <input type="number" min="1" step="256" value={params.max_tokens} onChange={(e) => setParam("max_tokens", e.target.value)} />
            </Field>
          </div>
          <button type="button" className="btn ghost sm" onClick={() => setParams(DEFAULT_PARAMS)}>
            Về mặc định
          </button>
        </div>
      )}

      <div className="chat-log" ref={logRef} onScroll={onLogScroll}>
        {messages.length === 0 && <div className="muted center">Gửi câu hỏi đầu tiên để bắt đầu. Enter để gửi, Shift+Enter để xuống dòng.</div>}
        {messages.map((m, i) => (
          <Message key={i} message={m} live={streaming && i === messages.length - 1} />
        ))}
      </div>

      <div className="chat-input">
        <textarea rows={2} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={onKeyDown} placeholder="Nhập câu hỏi…" />
        {streaming ? (
          <button type="button" className="btn" onClick={() => abortRef.current?.abort()}>
            <Square size={16} /> Dừng
          </button>
        ) : (
          <button type="button" className="btn primary" onClick={send} disabled={!input.trim()}>
            <Send size={16} /> Gửi
          </button>
        )}
      </div>
    </div>
  );
}
