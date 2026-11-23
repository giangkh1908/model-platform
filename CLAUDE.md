# Model Platform

Web local để tải, bật/tắt và thử các model AI chạy trong WSL (GPU 12 GB, mỗi lúc một model ở cổng 8000).

- Backend: FastAPI trong `backend/`, chạy bằng `platform_server.py` (cổng 8100). Shortcut `Model Platform.lnk` (cả trong project và trên Desktop) gọi `pythonw.exe platform_server.py --open --dev`: chạy cả backend và Vite dev server cổng 5173, mở trình duyệt tới 5173, không hiện terminal. Log ghi vào `data/platform.log`. Đóng hết tab giao diện thì server tự tắt sau ~5 s (`backend/presence.py`, tắt cả model và Vite). Khi cần thì tắt tay bằng Task Manager (tiến trình `pythonw.exe` chạy `platform_server.py`).
- Frontend: React + Vite trong `frontend/`. `npm run build` tạo `frontend/dist`, backend phục vụ thư mục này.
- Phát triển: `python platform_server.py --reload` và `cd frontend && npm run dev` (http://127.0.0.1:5173, proxy `/api`).
- Kiểm tra trước khi commit (CI chạy đúng các lệnh này, `.github/workflows/ci.yml`): `pip install -r requirements-dev.txt`, `ruff check .`, `pytest`; `cd frontend && npm run lint && npm run build`. Test chạy trên thư mục data tạm (`MODEL_PLATFORM_DATA`), không cần WSL/GPU.
- API chia theo router trong `backend/api/` (system, models, playground, finetune); `backend/app.py` chỉ dựng app, middleware, lifespan. Đổi route thì cập nhật `tests/routes.txt`.
- Model registry: `backend/model_registry.py` (interface `ModelRegistry`, MLflow), xuất bản job ở `backend/finetune/publish.py`. MLflow server chạy bằng `docker compose up -d`, file model trên Cloudflare R2. Prompt không đóng gói vào registry, chỉ ghi tag `<tên>@v<N>`.
- Giao diện và thông báo lỗi viết bằng tiếng Việt.
- Bí mật (vd. `HF_TOKEN`) để trong `.env` ở gốc project, nạp trong `backend/config.py`; biến Windows được ưu tiên; mẫu ở `.env.example`.
- Phạm vi: repo này chỉ lo finetune, thử model và eval. Thuật toán finetune sẽ tách sang repo `runtime-platform`; phục vụ người dùng (endpoint, RAG, Langfuse) thuộc repo `serving-platform`, không làm ở đây.

## Quy tắc code (theo plugin code-simplifier)

- Không đổi hành vi khi dọn code.
- JS/React: ES modules, import có đuôi `.js`/`.jsx`; hàm cấp cao nhất và component dùng `function`, arrow function chỉ dùng cho callback ngắn.
- Không lồng toán tử ba ngôi; nhiều nhánh thì dùng `if/else` hoặc `switch`.
- Chọn code rõ ràng hơn code ngắn; không viết one-liner dày đặc.
- Không lặp logic: hàm dùng chung đặt ở `frontend/src/lib.js` (JS) hoặc `backend/core.py` (Python).
- Python: có type hint cho hàm cấp cao nhất; lỗi người dùng thì ném `ValueError` (app trả về 400).
- Chỉ viết comment khi cần giải thích *tại sao*, không mô tả lại code.
