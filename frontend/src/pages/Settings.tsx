import { useEffect, useState } from "react";
import { CheckCircle2, KeyRound, RefreshCw, XCircle } from "lucide-react";
import { getConfig, getRuntimes, getSettings } from "../api/system.ts";
import { Card, Chip, CopyButton, Spinner } from "../components/ui.tsx";
import { endpointOf } from "../config/runtime.ts";
import { useApp } from "../store.tsx";
import type { PlatformConfig, RuntimeInfo, Settings as SettingsInfo, Status } from "../types/api.ts";

function TokenCard() {
  const [info, setInfo] = useState<SettingsInfo | null>(null);

  useEffect(() => {
    getSettings()
      .then(setInfo)
      .catch(() => setInfo(null));
  }, []);

  return (
    <Card title="Token Hugging Face">
      <p className="muted small">
        Cần cho model gated (Llama, Gemma…). Tạo token loại "Read" tại huggingface.co/settings/tokens. Điền <code>HF_TOKEN</code> vào file
        <code>.env</code> ở thư mục project (mẫu trong <code>.env.example</code>) rồi khởi động lại nền tảng; biến môi trường <code>HF_TOKEN</code> của
        Windows (nếu có) được ưu tiên hơn. Token không lưu trong thư mục data.
      </p>
      <div className="row wrap">
        <KeyRound size={16} />
        {info?.has_token ? (
          <>
            <Chip kind="ok">Đã đặt {info.token_masked}</Chip>
            {info.hf_user && <span className="small">Tài khoản: {info.hf_user}</span>}
          </>
        ) : (
          <Chip>Chưa đặt HF_TOKEN</Chip>
        )}
      </div>
    </Card>
  );
}

function RuntimesCard() {
  const [runtimes, setRuntimes] = useState<RuntimeInfo[] | null>(null);
  const [loading, setLoading] = useState(false);

  async function check() {
    setLoading(true);
    setRuntimes(await getRuntimes().catch((): RuntimeInfo[] => []));
    setLoading(false);
  }

  useEffect(() => {
    check();
  }, []);

  return (
    <Card
      title="Runtime trong WSL"
      actions={
        <button type="button" className="btn ghost sm" onClick={check} disabled={loading}>
          {loading ? <Spinner /> : <RefreshCw size={14} />} Kiểm tra lại
        </button>
      }
    >
      {!runtimes && <div className="muted">Đang kiểm tra (có thể phải bật WSL)…</div>}
      <div className="list">
        {runtimes?.map((r) => (
          <div key={r.id} className="list-row">
            {r.installed ? <CheckCircle2 size={18} className="ok-text" /> : <XCircle size={18} className="error-text" />}
            <div className="grow">
              <div className="strong">{r.name}</div>
              <code className="small break">{r.path}</code>
            </div>
            <Chip kind={r.installed ? "ok" : "bad"}>{r.installed ? "Đã cài" : "Chưa cài"}</Chip>
          </div>
        ))}
      </div>
    </Card>
  );
}

function exampleBody(runtime: string | null, mode: string | null, modelId: string | null): Record<string, unknown> {
  if (runtime === "decider") {
    return { state: "Help! My payouts have been failing for 3 days!", questions: { urgent: { type: "noul", instructions: "Is this urgent?" } } };
  }
  if (mode === "embed") return { model: modelId, input: ["xin chào", "hello"] };
  if (mode === "rerank") return { model: modelId, query: "giảm nhiệt GPU", documents: ["thay keo tản nhiệt", "nấu phở"] };
  return { model: modelId, messages: [{ role: "user", content: "Xin chào" }], max_tokens: 256 };
}

function curlExample(status: Status): string {
  const { model_url: baseUrl, runtime, mode, current: modelId } = status;
  const url = baseUrl + endpointOf(runtime, mode);
  const body = JSON.stringify(exampleBody(runtime, mode, modelId));
  return `curl ${url} -H "content-type: application/json" -d '${body}'`;
}

function ApiCard() {
  const { status } = useApp();
  const example = status?.state === "running" ? curlExample(status) : null;
  return (
    <Card title="Gọi model từ chương trình khác" className="span2">
      <p className="muted small">
        Model đang chạy phục vụ ở {status?.model_url ?? "http://127.0.0.1:8000"}. vLLM và llama.cpp dùng API kiểu OpenAI, nên các thư viện
        OpenAI chỉ cần đổi <code>base_url</code> thành <code>{(status?.model_url ?? "") + "/v1"}</code>.
      </p>
      {example ? (
        <div className="code-block">
          <pre className="code">{example}</pre>
          <CopyButton text={example} />
        </div>
      ) : (
        <div className="muted">Bật một model để xem lệnh mẫu.</div>
      )}
    </Card>
  );
}

function ConfigCard() {
  const [config, setConfig] = useState<PlatformConfig | null>(null);
  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch(() => {});
  }, []);
  if (!config) return null;
  const rows = [
    ["Bảng điều khiển", config.panel],
    ["Địa chỉ model", config.model_url],
    ["WSL distro", config.distro],
    ["Thư mục model", config.models_dir],
    ["VRAM", `${config.vram_gb} GB`],
  ];
  return (
    <Card title="Cấu hình" className="span2">
      <dl className="kv">
        {rows.map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>
              <code>{value}</code>
            </dd>
          </div>
        ))}
      </dl>
      <p className="muted small">Sửa trong backend/config.py rồi khởi động lại nền tảng.</p>
    </Card>
  );
}

export default function Settings() {
  return (
    <div className="grid">
      <TokenCard />
      <RuntimesCard />
      <ApiCard />
      <ConfigCard />
    </div>
  );
}
