import { FlaskConical, Loader } from "lucide-react";
import { Card, Chip, Empty } from "../components/ui.jsx";
import { MODE, RT, modeOf } from "../lib.js";
import { useModelActions } from "../modelActions.js";
import Chat from "../playground/Chat.jsx";
import Decide from "../playground/Decide.jsx";
import Embed from "../playground/Embed.jsx";
import ImageGen from "../playground/ImageGen.jsx";
import Rerank from "../playground/Rerank.jsx";
import Speak from "../playground/Speak.jsx";
import { useApp } from "../store.jsx";

const PANELS = { chat: Chat, decide: Decide, embed: Embed, rerank: Rerank, image: ImageGen, tts: Speak };

function NotRunning() {
  const { status, busy } = useApp();
  const { start } = useModelActions();
  const ready = status.models.filter((m) => m.status === "ready");

  if (status.state === "starting") {
    return (
      <Card>
        <Empty icon={Loader} title={`Đang khởi động ${status.current || ""}…`}>
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
            {m.id} <span className="muted small">· {MODE[modeOf(m)]}</span>
          </button>
        ))}
      </div>
    </Card>
  );
}

export default function Playground() {
  const { status } = useApp();
  if (!status) return <div className="muted">Đang tải…</div>;
  if (status.state !== "running") return <NotRunning />;

  const Panel = PANELS[status.mode] || Chat;
  return (
    <Card
      title={status.current}
      actions={
        <div className="row">
          <Chip>{RT[status.runtime] || "?"}</Chip>
          <Chip>{MODE[status.mode] || status.mode}</Chip>
        </div>
      }
    >
      <Panel key={status.current} modelId={status.current} />
    </Card>
  );
}
