import { useState } from "react";
import { Plus, Sparkles, X } from "lucide-react";
import { ask as askDecider } from "../api/playground.ts";
import { Field, ScoreBars, Spinner } from "../components/ui.tsx";
import { loadLocal, saveLocal } from "../lib/browser.ts";
import { errorText, percent } from "../lib/format.ts";
import type { DecideAnswer, DecideResponse } from "../types/api.ts";

type QuestionType = "noul" | "choice" | "score";

/** Một câu hỏi đang soạn: lựa chọn gõ thành một chuỗi, cách nhau bằng dấu phẩy. */
interface DraftQuestion {
  type: QuestionType;
  text: string;
  options: string;
}

interface Preset {
  name: string;
  state: string;
  questions: DraftQuestion[];
}

interface AnswerView {
  title: string;
  detail: string;
  rows: [string, number][];
  top: string;
}

const TYPES: { id: QuestionType; label: string }[] = [
  { id: "noul", label: "Có/không" },
  { id: "choice", label: "Chọn 1" },
  { id: "score", label: "Thang điểm" },
];

// Mẫu lấy từ examples/test_decider.py
const PRESETS: Preset[] = [
  {
    name: "Hỗ trợ khách hàng",
    state: "Tôi đã chuyển khoản 3 lần mà đơn hàng vẫn báo chưa thanh toán. Làm ăn kiểu gì vậy?!",
    questions: [
      { type: "choice", text: "Bộ phận nào nên xử lý?", options: "thanh toán, kỹ thuật, bán hàng, vận chuyển" },
      { type: "noul", text: "Khách có đang rất gấp hoặc bức xúc không?", options: "" },
      { type: "score", text: "Mức độ bực tức của khách?", options: "bình tĩnh, khó chịu, rất tức giận" },
    ],
  },
  {
    name: "Chọn công cụ cho agent",
    state: "Người dùng: Thời tiết ở Hà Nội ngày mai thế nào?",
    questions: [{ type: "choice", text: "Agent nên gọi công cụ nào?", options: "tìm kiếm web, dự báo thời tiết, máy tính, lịch" }],
  },
  {
    name: "Định tuyến model",
    state: "Viết cho tôi một bài phân tích chi tiết 3000 chữ về tác động của AI tới thị trường lao động Việt Nam, có số liệu và trích dẫn.",
    questions: [{ type: "choice", text: "Nên gửi yêu cầu này cho model nào?", options: "model nhỏ (nhanh, rẻ), model lớn (mạnh, đắt)" }],
  },
  {
    name: "Kiểm duyệt nội dung",
    state: "CHÚC MỪNG! Bạn đã trúng iPhone 17. Bấm vào link bit.ly/abc123 để nhận quà trong 24h!",
    questions: [
      { type: "noul", text: "Đây có phải tin nhắn lừa đảo/spam không?", options: "" },
      { type: "noul", text: "Tin nhắn có chứa đe doạ bạo lực không?", options: "" },
    ],
  },
  {
    name: "Cảm xúc đánh giá",
    state: "Pin trâu, màn đẹp nhưng camera chụp tối hơi nhoè. Nhìn chung vẫn đáng tiền.",
    questions: [{ type: "score", text: "Đánh giá chung của người viết?", options: "rất tệ, tệ, trung bình, tốt, rất tốt" }],
  },
  {
    name: "Áp dụng chính sách",
    state: "Chính sách: Hoàn tiền nếu yêu cầu trong vòng 30 ngày và sản phẩm chưa mở hộp.\nTrường hợp: Khách mua 12 ngày trước, đã mở hộp dùng thử, muốn trả hàng.",
    questions: [{ type: "noul", text: "Theo chính sách, khách có được hoàn tiền không?", options: "" }],
  },
];

function toQuestion({ type, text, options }: DraftQuestion): Record<string, unknown> {
  const list = options.split(",").map((s) => s.trim()).filter(Boolean);
  if (type === "noul") return { type, instructions: text };
  if (type === "choice") return { type, instructions: text, criteria: Object.fromEntries(list.map((o) => [o, null])) };
  return { type, instructions: text, criteria: list };
}

function describeAnswer(answer: DecideAnswer): AnswerView {
  if (answer.type === "noul") {
    const top = answer.noul >= 0.5 ? "có" : "không";
    return {
      title: top === "có" ? "Có" : "Không",
      detail: `xác suất "có" ${percent(answer.noul)}`,
      rows: [
        ["có", answer.noul],
        ["không", 1 - answer.noul],
      ],
      top,
    };
  }
  if (answer.type === "choice") {
    return {
      title: answer.choice,
      detail: `độ tin cậy ${percent(answer.confidence)}`,
      rows: Object.entries(answer.probabilities),
      top: answer.choice,
    };
  }
  const rows = Object.entries(answer.probabilities).map(([index, value]): [string, number] => [answer.legend?.[index] ?? index, value]);
  const top = rows.reduce((best, row) => (row[1] > best[1] ? row : best), rows[0])[0];
  return { title: top, detail: `điểm ${answer.score.toFixed(2)} · độ tin cậy ${percent(answer.confidence)}`, rows, top };
}

interface QuestionRowProps {
  question: DraftQuestion;
  onChange: (patch: Partial<DraftQuestion>) => void;
  onRemove: () => void;
}

function QuestionRow({ question, onChange, onRemove }: QuestionRowProps) {
  const needsOptions = question.type !== "noul";
  return (
    <div className="question">
      <select value={question.type} onChange={(e) => onChange({ type: e.target.value as QuestionType })}>
        {TYPES.map((t) => (
          <option key={t.id} value={t.id}>
            {t.label}
          </option>
        ))}
      </select>
      <input value={question.text} onChange={(e) => onChange({ text: e.target.value })} placeholder="Câu hỏi" />
      <input
        value={question.options}
        onChange={(e) => onChange({ options: e.target.value })}
        disabled={!needsOptions}
        placeholder={needsOptions ? "Lựa chọn, cách nhau bằng dấu phẩy" : "Không cần lựa chọn"}
      />
      <button type="button" className="icon-btn" onClick={onRemove} title="Xoá câu hỏi">
        <X size={16} />
      </button>
    </div>
  );
}

export default function Decide() {
  const [state, setState] = useState(() => loadLocal("decide:state", PRESETS[0].state));
  const [questions, setQuestions] = useState<DraftQuestion[]>(() => loadLocal("decide:questions", PRESETS[0].questions));
  const [result, setResult] = useState<{ response: DecideResponse; labels: Record<string, string> } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function applyPreset(name: string) {
    const preset = PRESETS.find((p) => p.name === name);
    if (!preset) return;
    setState(preset.state);
    setQuestions(preset.questions);
    setResult(null);
  }

  function updateQuestion(index: number, patch: Partial<DraftQuestion>) {
    setQuestions((list) => list.map((q, i) => (i === index ? { ...q, ...patch } : q)));
  }

  async function ask() {
    const filled = questions.filter((q) => q.text.trim());
    if (!filled.length) {
      setError("Thêm ít nhất một câu hỏi.");
      return;
    }
    const payload = Object.fromEntries(filled.map((q, i) => [`q${i + 1}`, toQuestion(q)]));
    const labels = Object.fromEntries(filled.map((q, i) => [`q${i + 1}`, q.text]));
    saveLocal("decide:state", state);
    saveLocal("decide:questions", questions);
    setLoading(true);
    setError("");
    try {
      const response = await askDecider(state, payload);
      setResult({ response, labels });
    } catch (e) {
      setError(errorText(e));
    }
    setLoading(false);
  }

  return (
    <div className="stack">
      <div className="row between wrap">
        <span className="muted small">
          Có/không: xác suất đúng · Chọn 1: chọn một lựa chọn · Thang điểm: chọn mức từ thấp đến cao.
        </span>
        <label className="row small">
          <Sparkles size={14} /> Mẫu
          <select value="" onChange={(e) => applyPreset(e.target.value)} className="auto">
            <option value="">Chọn mẫu…</option>
            {PRESETS.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
      </div>
      <Field label="Văn bản cần đánh giá">
        <textarea rows={3} value={state} onChange={(e) => setState(e.target.value)} />
      </Field>
      <div className="questions">
        {questions.map((q, i) => (
          <QuestionRow
            key={i}
            question={q}
            onChange={(patch) => updateQuestion(i, patch)}
            onRemove={() => setQuestions((list) => list.filter((_, j) => j !== i))}
          />
        ))}
      </div>
      <div className="row wrap">
        <button type="button" className="btn" onClick={() => setQuestions([...questions, { type: "noul", text: "", options: "" }])}>
          <Plus size={16} /> Thêm câu hỏi
        </button>
        <button type="button" className="btn primary" onClick={ask} disabled={loading}>
          {loading && <Spinner />} Hỏi
        </button>
        {result && (
          <span className="muted small">
            {result.response._ms} ms · {result.response.usage?.input_tokens ?? "?"} token
          </span>
        )}
        {error && <span className="error-text small">{error}</span>}
      </div>
      {result && (
        <div className="answers">
          {Object.entries(result.response.answers).map(([key, answer]) => {
            const view = describeAnswer(answer);
            return (
              <div key={key} className="answer">
                <div className="muted small">{result.labels[key] ?? key}</div>
                <div className="answer-head">
                  <b>{view.title}</b>
                  <span className="muted small">{view.detail}</span>
                </div>
                <ScoreBars rows={view.rows} top={view.top} format={(v) => percent(v)} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
