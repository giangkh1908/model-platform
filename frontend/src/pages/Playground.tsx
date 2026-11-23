import type { ComponentType } from "react";
import { FlaskConical, Loader } from "lucide-react";
import { Card, Chip, Empty } from "../components/ui.tsx";
import { modeName, modeOf, runtimeName } from "../config/runtime.ts";
import { useModelActions } from "../modelActions.ts";
import Chat from "../playground/Chat.tsx";
import Decide from "../playground/Decide.tsx";
import Embed from "../playground/Embed.tsx";
import ImageGen from "../playground/ImageGen.tsx";
import Rerank from "../playground/Rerank.tsx";
import Speak from "../playground/Speak.tsx";
import { useApp } from "../store.tsx";
import type { Status } from "../types/api.ts";

/** Props chung của mọi khung thử model. */
export interface PanelProps {
  modelId: string;
}

/** Mỗi chế độ chạy model một khung thử. Thêm chế độ mới: thêm một mục ở đây. */
const PANELS: Record<string, ComponentType<PanelProps>> = { chat: Chat, decide: Decide, embed: Embed, rerank: Rerank, image: ImageGen, tts: Speak };

function NotRunning({ status }: { status: Status }) {
  const { busy } = useApp();
  const { start } = useModelActions();
  const ready = status.models.filter((m) => m.status === "ready");

  if (status.state === "starting") {
    return (
      <Card>
        <Empty icon={Loader} title={`Đang khởi động ${status.current ?? ""}…`}>
          Đợi model nạp xong vào GPU. Tiến trình hiện ở trang Log.
        </Empty>
      </Card>
    );
  }
  return (
    <Card>
      <Empty icon={FlaskConical} title="Chưa có model nào chạy">
        Bật một model để thử:
      </Empty>
      <div className="row wrap center">
        {ready.map((m) => (
          <button key={m.id} type="button" className="btn" disabled={busy} onClick={() => start(m)}>
            {m.id} <span className="muted small">· {modeName(modeOf(m))}</span>
          </button>
        ))}
      </div>
    </Card>
  );
}

export default function Playground() {
  const { status } = useApp();
  if (!status) return <div className="muted">Đang tải…</div>;
  if (status.state !== "running" || !status.current) return <NotRunning status={status} />;

  const Panel = PANELS[status.mode ?? "chat"] ?? Chat;
  return (
    <Card
      title={status.current}
      actions={
        <div className="row">
          <Chip>{status.runtime ? runtimeName(status.runtime) : "?"}</Chip>
          <Chip>{status.mode ? modeName(status.mode) : "?"}</Chip>
        </div>
      }
    >
      <Panel key={status.current} modelId={status.current} />
    </Card>
  );
}
