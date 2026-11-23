// Giao diện riêng của loại fine-tune "decider" (intent + gate): cài đặt interface TaskUI (finetune/tasks.ts).
import { useState } from "react";
import { Chip } from "../components/ui.tsx";
import { formatMetric } from "../lib/format.ts";
import type { EvalModelResult, EvalReportData, ThresholdRates } from "../types/api.ts";
import LineChart from "./LineChart.tsx";

// Xếp hạng: accuracy cao hơn trước, bằng nhau thì macro-F1.
function ranked(models: EvalModelResult[]): EvalModelResult[] {
  return [...models].sort((a, b) => b.summary.acc - a.summary.acc || b.summary.macro_f1 - a.summary.macro_f1);
}

function KindChip({ kind }: { kind: EvalModelResult["kind"] }) {
  if (kind === "base") return <Chip>model gốc</Chip>;
  return <Chip kind="ok">adapter</Chip>;
}

function atThreshold(model: EvalModelResult, key: keyof ThresholdRates, unit: string): string {
  if (!model.at_threshold) return "–";
  return formatMetric(model.at_threshold[key], unit);
}

// Các cột tại ngưỡng tự động / chuyển người: [khoá trong at_threshold, tiêu đề, đơn vị, giải thích]
const THRESHOLD_COLUMNS: [keyof ThresholdRates, string, string, string][] = [
  ["tp", "TP", "", "Tự động và đúng"],
  ["fp", "FP", "", "Tự động nhưng sai: lỗi lọt ra ngoài"],
  ["fn", "FN", "", "Chuyển người dù model đã đúng"],
  ["tn", "TN", "", "Chuyển người và chặn được câu sai"],
  ["precision", "Precision", "%", "TP / (TP + FP): phần tự động đúng bao nhiêu"],
  ["recall", "Recall", "%", "TP / (TP + FN): câu đúng được tự động bao nhiêu"],
  ["f1", "F1", "%", "Trung bình điều hoà của precision và recall"],
  ["coverage", "Coverage", "%", "(TP + FP) / tổng: tự động được bao nhiêu câu"],
];

function Leaderboard({ models, selected, onSelect }: { models: EvalModelResult[]; selected: string; onSelect: (key: string) => void }) {
  return (
    <div className="stack">
      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>#</th>
              <th>Model</th>
              <th className="num" title="Tỉ lệ câu chọn đúng nhãn, trên toàn bộ tập">
                Accuracy
              </th>
              <th className="num" title="Trung bình F1 của các nhãn: nhãn ít câu cũng được tính ngang nhãn nhiều câu">
                Macro-F1
              </th>
              <th className="num" title="Độ lệch giữa độ tin cậy model báo và tỉ lệ đúng thực tế">
                ECE
              </th>
              <th className="num" title="Chọn trên val lúc train: độ tin cậy ≥ ngưỡng thì tự động">
                Ngưỡng
              </th>
              {THRESHOLD_COLUMNS.map(([key, label, , hint]) => (
                <th key={key} className="num" title={hint}>
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((m, i) => (
              <tr key={m.key} className={`clickable ${m.key === selected ? "selected" : ""}`} onClick={() => onSelect(m.key)}>
                <td className="rank">{i + 1}</td>
                <td>
                  <span className="row nowrap">
                    <b>{m.name}</b>
                    <KindChip kind={m.kind} />
                  </span>
                </td>
                <td className="num strong">{formatMetric(m.summary.acc, "%")}</td>
                <td className="num">{formatMetric(m.summary.macro_f1, "%")}</td>
                <td className="num">{formatMetric(m.summary.ece, "%")}</td>
                <td className="num">{m.threshold == null ? "–" : formatMetric(m.threshold)}</td>
                {THRESHOLD_COLUMNS.map(([key, , unit]) => (
                  <td key={key} className="num">
                    {atThreshold(m, key, unit)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        TP / FP / FN / TN tính tại ngưỡng của từng adapter, với "dương tính" là model trả lời đúng: TP tự động và đúng, FP tự động nhưng sai
        (lỗi lọt ra ngoài), FN chuyển người dù đã đúng, TN chuyển người và chặn được câu sai. Model gốc chưa có ngưỡng nên để trống. Rê chuột
        lên tiêu đề cột để xem giải thích.
      </p>
    </div>
  );
}

// F1 từng nhãn của mọi model, sắp theo model đang chọn (yếu nhất trước).
function LabelMatrix({ models, selectedModel }: { models: EvalModelResult[]; selectedModel: EvalModelResult }) {
  const byModel = new Map(models.map((m) => [m.key, new Map((m.per_label ?? []).map((r) => [r.label, r]))]));
  const rows = (selectedModel.per_label ?? []).filter((r) => r.n).sort((a, b) => a.f1 - b.f1);
  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th>Nhãn</th>
            <th className="num">Số câu</th>
            {models.map((m) => (
              <th key={m.key} className={`num ${m.key === selectedModel.key ? "strong" : ""}`}>
                F1 · {m.name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <td>
                <code>{r.label}</code>
              </td>
              <td className="num">{r.n}</td>
              {models.map((m) => (
                <td key={m.key} className={`num ${m.key === selectedModel.key ? "strong" : "muted"}`}>
                  {formatMetric(byModel.get(m.key)?.get(r.label)?.f1, "%")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelErrors({ model }: { model: EvalModelResult }) {
  const confusions = model.confusions ?? [];
  const mistakes = model.mistakes ?? [];
  return (
    <div className="stack">
      {confusions.length > 0 && (
        <>
          <div className="steps-title">Cặp nhãn hay nhầm · {model.name}</div>
          <p className="small muted">Nhầm nhiều giữa hai nhãn: thêm mô tả phân biệt cho chúng hoặc thêm dữ liệu.</p>
          <div className="row wrap">
            {confusions.map((c) => (
              <Chip key={`${c.gold}>${c.pred}`} title="nhãn đúng → model chọn">
                {c.gold} → {c.pred} · {c.count}
              </Chip>
            ))}
          </div>
        </>
      )}
      {mistakes.length > 0 && (
        <>
          <div className="steps-title">Câu sai mà model tự tin nhất · {model.name}</div>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Câu</th>
                  <th>Nhãn đúng</th>
                  <th>Model chọn</th>
                  <th className="num">Độ tin cậy</th>
                </tr>
              </thead>
              <tbody>
                {mistakes.map((m, i) => (
                  <tr key={i}>
                    <td className="text">{m.text}</td>
                    <td>
                      <code>{m.gold}</code>
                    </td>
                    <td>
                      <code>{m.pred}</code>
                    </td>
                    <td className="num">{formatMetric(m.confidence, "%")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted">Câu sai với độ tin cậy cao thường là nhãn sai trong dữ liệu: nên kiểm tra lại.</p>
        </>
      )}
    </div>
  );
}

// Mỗi gate: precision / recall / F1 của trường hợp "có" và ECE, của từng model.
function GateTable({ models, selectedModel }: { models: EvalModelResult[]; selectedModel: EvalModelResult }) {
  const gates = selectedModel.gates ?? {};
  const keys = Object.keys(gates);
  if (!keys.length) return null;
  return (
    <>
      <div className="steps-title">Gate (câu có/không)</div>
      <p className="small muted">Precision / recall / F1 tính cho trường hợp &quot;có&quot; (dính gate), ngưỡng 50%.</p>
      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>Gate</th>
              <th className="num">Câu dính</th>
              {models.map((m) => (
                <th key={m.key} className={`num ${m.key === selectedModel.key ? "strong" : ""}`}>
                  P / R / F1 · {m.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {keys.map((key) => (
              <tr key={key}>
                <td>
                  <code>{key}</code>
                </td>
                <td className="num">{gates[key].positives}</td>
                {models.map((m) => {
                  const g = m.gates?.[key];
                  return (
                    <td key={m.key} className={`num ${m.key === selectedModel.key ? "strong" : "muted"}`}>
                      {g ? `${formatMetric(g.precision, "%")} / ${formatMetric(g.recall, "%")} / ${formatMetric(g.f1, "%")}` : "–"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {keys
        .filter((key) => gates[key].mistakes.length)
        .map((key) => (
          <details key={key}>
            <summary className="small muted">
              Câu sai của gate {key} · {selectedModel.name} ({gates[key].mistakes.length})
            </summary>
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th>Câu</th>
                    <th>Đúng</th>
                    <th className="num">P(có)</th>
                  </tr>
                </thead>
                <tbody>
                  {gates[key].mistakes.map((m, i) => (
                    <tr key={i}>
                      <td className="text">{m.text}</td>
                      <td>
                        <code>{m.gold === "true" ? "có" : "không"}</code>
                      </td>
                      <td className="num">{formatMetric(m.p_true, "%")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        ))}
    </>
  );
}

export function EvalReport({ report }: { report: EvalReportData }) {
  const models = ranked(report.models);
  const [selected, setSelected] = useState(models[0].key);
  const selectedModel = models.find((m) => m.key === selected) || models[0];
  const hasIntent = Boolean(selectedModel.per_label);

  return (
    <div className="stack">
      <p className="small muted">
        {report.n} câu golden. Xếp theo accuracy của intent (bộ chỉ có gate: trung bình các gate). Bấm một dòng để xem chi tiết model đó.
      </p>
      <Leaderboard models={models} selected={selectedModel.key} onSelect={setSelected} />
      {hasIntent && (
        <>
          <LineChart
            title="Intent: precision theo coverage"
            hint="Mỗi điểm là một ngưỡng: tự động bao nhiêu câu thì phần tự động đúng bao nhiêu. Đường càng cao càng tốt."
            xLabel="Coverage"
            formatX={(x) => formatMetric(x, "%")}
            unit="%"
            dots={false}
            series={models.map((m) => ({ label: m.name, points: (m.curve ?? []).map((c): [number, number] => [c.coverage, c.precision]) })).filter((x) => x.points.length)}
          />
          <div className="steps-title">F1 từng intent (yếu nhất trước, theo {selectedModel.name})</div>
          <LabelMatrix models={models.filter((m) => m.per_label)} selectedModel={selectedModel} />
          <ModelErrors model={selectedModel} />
        </>
      )}
      <GateTable models={models} selectedModel={selectedModel} />
    </div>
  );
}
