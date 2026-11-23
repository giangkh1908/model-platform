"""Model Platform: web local để tải, bật/tắt và thử các model AI chạy trong WSL.

Chạy:      python platform_server.py --open --dev   (bật cả backend và giao diện dev; mở http://127.0.0.1:5173)
           python platform_server.py --open         (chỉ backend, giao diện lấy từ frontend/dist)
Đóng cửa sổ console hoặc Ctrl+C để tắt tất cả.
Mở:        http://127.0.0.1:8100
Phát triển: python platform_server.py --reload   + trong frontend/: npm run dev  (http://127.0.0.1:5173)

Backend: FastAPI (thư mục backend/). Giao diện: React + Vite (thư mục frontend/, bản build ở frontend/dist).

Runtime hỗ trợ:
  decider   Strands Decider (model ra quyết định)  -> strands-decider serve
  vllm      model chat dạng safetensors             -> vllm serve       (API kiểu OpenAI)
  llamacpp  model chat dạng GGUF                    -> llama-server     (API kiểu OpenAI)
Mọi model đều phục vụ ở http://127.0.0.1:8000, mỗi lúc chỉ chạy một model (GPU 12GB).
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRONTEND = HERE / "frontend"
VITE_ENTRY = FRONTEND / "node_modules" / "vite" / "bin" / "vite.js"
sys.path.insert(0, str(HERE))

from backend.config import DATA, NO_WINDOW, PANEL_HOST, PANEL_PORT, ROOT_PID_ENV  # noqa: E402

DEV_PORT = 5173


def port_busy(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def prepare_output() -> None:
    # Chạy bằng pythonw (không console) thì sys.stdout/sys.stderr là None; uvicorn sẽ lỗi,
    # nên chuyển log sang file trong data/ để vừa chạy được vừa có log khi cần xem.
    if sys.stdout is None or sys.stderr is None:
        log = open(DATA / "platform.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = log
        sys.stderr = log
    # Console Windows có thể dùng cp1252; print tiếng Việt sẽ làm server chết trước khi chạy.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")


def start_frontend_dev() -> subprocess.Popen | None:
    if not VITE_ENTRY.exists():
        print("Chưa cài thư viện giao diện. Chạy trong frontend/: npm install")
        return None
    if port_busy(PANEL_HOST, DEV_PORT):
        print(f"Cổng {DEV_PORT} đang được dùng: giao diện dev có thể đã chạy sẵn.")
        return None
    try:
        # Truyền rõ stdout/stderr: khi chạy bằng pythonw, handle thừa kế có thể không hợp lệ và Vite thoát ngay.
        # NO_WINDOW: node là ứng dụng console, nếu không có cờ này Windows sẽ mở thêm một cửa sổ terminal.
        return subprocess.Popen(["node", str(VITE_ENTRY)], cwd=FRONTEND, stdin=subprocess.DEVNULL,
                                stdout=sys.stdout, stderr=sys.stderr, creationflags=NO_WINDOW)
    except OSError as exc:
        print(f"Không chạy được giao diện dev (cần Node.js trong PATH): {exc}")
        return None


def stop_process(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    prepare_output()
    ap = argparse.ArgumentParser(description="Model Platform")
    ap.add_argument("--open", action="store_true", help="mở trình duyệt sau khi chạy")
    ap.add_argument("--dev", action="store_true", help="chạy thêm giao diện dev (Vite, cổng 5173)")
    ap.add_argument("--reload", action="store_true", help="tự nạp lại khi sửa code backend (phát triển)")
    args = ap.parse_args()
    url = f"http://{PANEL_HOST}:{PANEL_PORT}"
    os.environ[ROOT_PID_ENV] = str(os.getpid())  # worker của --reload kế thừa biến này để tắt cả cây tiến trình

    if port_busy(PANEL_HOST, PANEL_PORT):
        print(f"Cổng {PANEL_PORT} đang được dùng: Model Platform đã chạy ở cửa sổ khác. Mở {url}")
        if args.open:
            webbrowser.open(url)
        return

    import uvicorn

    frontend_proc = start_frontend_dev() if args.dev else None
    open_url = f"http://{PANEL_HOST}:{DEV_PORT}" if frontend_proc else url

    print(f"Model Platform đang chạy ở {url}  (Ctrl+C hoặc đóng cửa sổ này để tắt)")
    if frontend_proc:
        print(f"Giao diện dev đang chạy ở http://{PANEL_HOST}:{DEV_PORT}")
    if args.open:
        threading.Timer(3.0 if frontend_proc else 1.0, webbrowser.open, [open_url]).start()
    os.chdir(HERE)
    try:
        uvicorn.run("backend.app:app", host=PANEL_HOST, port=PANEL_PORT, reload=args.reload,
                    reload_dirs=[str(HERE / "backend")] if args.reload else None,
                    log_level="warning", access_log=False)
    finally:
        if frontend_proc:
            stop_process(frontend_proc)


if __name__ == "__main__":
    main()
