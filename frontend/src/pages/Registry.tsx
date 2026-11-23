// Trang Registry: model đã xuất bản lên MLflow, các phiên bản và alias.
// champion = bản serving-platform đang dùng, challenger = bản đang thử. Xuất bản từ trang chi tiết job train.
import { useEffect, useState } from "react";
import { ExternalLink, Package, RefreshCw, X } from "lucide-react";
import * as registryApi from "../api/registry.ts";
import { Card, Chip, Empty, Spinner } from "../components/ui.tsx";
import { formatMetric, formatTime } from "../lib/format.ts";
import { useApp } from "../store.tsx";
import type { RegistryModel, RegistryStatus, RegistryVersion } from "../types/api.ts";

/** Cột chỉ số hiện trong bảng phiên bản (khoá là tên chỉ số trên MLflow). */
const METRICS: { key: string; label: string; unit: string }[] = [
  { key: "golden_acc", label: "Accuracy (eval)", unit: "%" },
  { key: "golden_macro_f1", label: "Macro-F1 (eval)", unit: "%" },
  { key: "golden_precision_at_threshold", label: "Precision tại ngưỡng", unit: "%" },
  { key: "final_threshold_threshold", label: "Ngưỡng", unit: "" },
];

function StatusCard({ status, onReload, loading }: { status: RegistryStatus | null; onReload: () => void; loading: boolean }) {
  let body = <Spinner />;
  if (status && !status.enabled) body = <Chip>Chưa cấu hình</Chip>;
  else if (status && status.online) body = <Chip kind="ok">Đang kết nối</Chip>;
  else if (status) body = <Chip kind="bad">Không kết nối được</Chip>;

  return (
    <Card
      title="Model registry (MLflow)"
      className="span2"
      actions={
        <button type="button" className="btn ghost sm" onClick={onReload} disabled={loading}>
          {loading ? <Spinner /> : <RefreshCw size={14} />} Tải lại
        </button>
      }
    >
      <div className="row wrap">
        {body}
        {status?.ui && (
          <a className="small" href={status.ui} target="_blank" rel="noreferrer">
            Mở giao diện MLflow <ExternalLink size={12} />
          </a>
        )}
      </div>
      {status?.message && <p className="muted small">{status.message}</p>}
      <p className="muted small">
        Xuất bản ở trang chi tiết job train (nút "Xuất bản"). Gán <b>champion</b> cho bản serving-platform sẽ dùng, <b>challenger</b> cho bản
        đang thử. Prompt không nằm trong gói: phiên bản system / intent / gate ghi ở tag, nội dung lấy từ kho prompt (Langfuse).
      </p>
    </Card>
  );
}

function VersionRow({ model, version, aliases, onChanged }: { model: string; version: RegistryVersion; aliases: string[]; onChanged: () => void }) {
  const { busy, run } = useApp();

  async function setAlias(alias: string) {
    const ok = await run(() => registryApi.setAlias(model, alias, version.version));
    if (ok) onChanged();
  }

  async function removeAlias(alias: string) {
    const ok = await run(() => registryApi.deleteAlias(model, alias));
    if (ok) onChanged();
  }

  const tags = version.tags;
  return (
    <tr>
      <td>
        <b>v{version.version}</b>
        <div className="muted small">{formatTime(version.created)}</div>
      </td>
      <td className="small">
        <div>{tags.build || "—"}</div>
        <div className="muted">từ {tags.base_model || "?"}</div>
      </td>
      <td className="small">
        <div>{tags["prompt.system"] || "—"}</div>
        <div className="muted">{tags["prompt.intents"] || ""}</div>
        <div className="muted">{tags["prompt.gates"] || ""}</div>
      </td>
      {METRICS.map((m) => (
        <td key={m.key} className="small">
          {version.metrics[m.key] == null ? "—" : formatMetric(version.metrics[m.key], m.unit)}
        </td>
      ))}
      <td>
        <div className="row wrap">
          {version.aliases.map((alias) => (
            <Chip key={alias} kind={alias === "champion" ? "ok" : ""}>
              {alias}
              <button type="button" className="btn ghost sm" disabled={busy} onClick={() => removeAlias(alias)} title={`Bỏ ${alias}`}>
                <X size={12} />
              </button>
            </Chip>
          ))}
          {aliases
            .filter((alias) => !version.aliases.includes(alias))
            .map((alias) => (
              <button key={alias} type="button" className="btn sm" disabled={busy} onClick={() => setAlias(alias)}>
                Gán {alias}
              </button>
            ))}
        </div>
      </td>
    </tr>
  );
}

function ModelCard({ model, aliases, onChanged }: { model: RegistryModel; aliases: string[]; onChanged: () => void }) {
  return (
    <Card title={model.name} className="span2">
      {model.description && <p className="muted small">{model.description}</p>}
      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>Phiên bản</th>
              <th>Build</th>
              <th>Prompt</th>
              {METRICS.map((m) => (
                <th key={m.key}>{m.label}</th>
              ))}
              <th>Alias</th>
            </tr>
          </thead>
          <tbody>
            {model.versions.map((v) => (
              <VersionRow key={v.version} model={model.name} version={v} aliases={aliases} onChanged={onChanged} />
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

export default function Registry() {
  const [status, setStatus] = useState<RegistryStatus | null>(null);
  const [models, setModels] = useState<RegistryModel[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const s = await registryApi.getRegistryStatus();
      setStatus(s);
      if (s.enabled && s.online) {
        setModels(await registryApi.listRegistryModels());
      } else {
        setModels(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  const aliases = status?.aliases ?? [];
  return (
    <div className="grid">
      <StatusCard status={status} onReload={load} loading={loading} />
      {error && <div className="notice bad span2">{error}</div>}
      {models && models.length === 0 && (
        <Card className="span2">
          <Empty icon={Package} title="Chưa có model nào được xuất bản">
            Train xong một job, vào trang chi tiết job rồi bấm "Xuất bản".
          </Empty>
        </Card>
      )}
      {models?.map((m) => (
        <ModelCard key={m.name} model={m} aliases={aliases} onChanged={load} />
      ))}
    </div>
  );
}
