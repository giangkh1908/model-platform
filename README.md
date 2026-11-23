# Model Platform

[![CI](https://github.com/giangkh1908/model-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/giangkh1908/model-platform/actions/workflows/ci.yml)

Web chạy trên máy để tải, bật/tắt, thử, fine-tune và eval các model AI chạy trong WSL (GPU RTX 3060 12 GB).

- Bảng điều khiển: http://127.0.0.1:8100 (bản dev: http://127.0.0.1:5173)
- Model đang chạy phục vụ ở http://127.0.0.1:8000 (mỗi lúc chỉ chạy một model)

## Vai trò trong hệ thống

Đây là một trong ba repo:

| Repo | Vai trò |
| --- | --- |
| **model-platform** (repo này) | "Phòng thí nghiệm": dataset, prompt, build, bấm train, thử model, eval |
| `runtime-platform` | Thuật toán fine-tune. Platform gọi sang qua [runtime protocol](docs/runtime-protocol.md) |
| `serving-platform` | Phục vụ người dùng thật: endpoint chuẩn, RAG, semantic search, Langfuse |

Repo này không phục vụ người dùng cuối và không có phần giám sát (Langfuse…), những việc đó thuộc `serving-platform`.

## Cần có

| Thành phần | Ghi chú |
| --- | --- |
| Windows + WSL (distro `Ubuntu`) | Đã cài sẵn runtime: Strands Decider, vLLM, llama.cpp |
| Python 3.10 trở lên | Đã thử với 3.12 |
| Node.js 18 trở lên | Cần để build hoặc sửa giao diện (CI dùng Node 24) |
| Driver NVIDIA | Cần lệnh `nvidia-smi` để hiện thông số GPU |

## Cài đặt (làm một lần)

Mở PowerShell ở thư mục gốc dự án:

```powershell
# 1. Thư viện Python cho backend
pip install -r requirements.txt

# 2. Build giao diện React
cd frontend
npm install
npm run build
cd ..

# 3. File bí mật (token…)
copy .env.example .env
```

Sau bước 2 sẽ có thư mục `frontend/dist`, backend phục vụ giao diện từ thư mục này.

## Token Hugging Face

Cần cho model gated (Llama, Gemma…). Tạo token loại *Read* tại huggingface.co/settings/tokens, rồi điền vào file `.env`:

```
HF_TOKEN=hf_xxxxxxxx
```

Khởi động lại nền tảng để nhận token mới. Kiểm tra ở trang **Cài đặt** → *Token Hugging Face*: phải hiện *Đã đặt hf_xx…xxxx* và tên tài khoản.

- `.env` chứa bí mật: không commit (đã có trong `.gitignore`), không gửi cho ai.
- Nếu Windows đã có biến môi trường `HF_TOKEN` (đặt bằng `setx`) thì biến đó được ưu tiên hơn `.env`.

## Khởi động

Project chỉ chạy khi bạn khởi động nó, không có gì chạy ngầm hay tự bật cùng Windows.

**Cách 1: bấm đúp `Model Platform.lnk`** (trong project hoặc trên Desktop). Shortcut chạy `pythonw.exe platform_server.py --open --dev`: bật backend và Vite dev server, mở trình duyệt tới cổng 5173, không hiện cửa sổ terminal. Log ghi vào `data/platform.log`.

**Cách 2: dòng lệnh** (ở thư mục gốc dự án)

```powershell
python platform_server.py --open     # tự mở trình duyệt
python platform_server.py            # không mở trình duyệt
```

Nếu đã có một bản đang chạy thì lệnh mới chỉ báo "cổng 8100 đang được dùng" rồi thoát.

Tài liệu API do FastAPI tự tạo: http://127.0.0.1:8100/docs

## Tắt

**Đóng hết tab giao diện là nền tảng tự tắt** sau khoảng 5 giây, kèm tắt model đang chạy trong WSL để trả VRAM và tắt Vite. Tải lại trang (F5) không làm tắt server. Còn việc trong hàng đợi fine-tune thì nền tảng không tự tắt, để train qua đêm không cần mở tab.

Cần tắt tay thì dùng Task Manager, kết thúc tiến trình `pythonw.exe` đang chạy `platform_server.py`.

Trên giao diện còn có:
- **Tắt model**: trả VRAM, server vẫn chạy.
- **Tắt hẳn WSL**: tắt model và toàn bộ WSL, trả RAM cho Windows.

## Dataset, Prompt, Build, Finetune, Eval

Các mục trên thanh bên, đi theo thứ tự:

1. **Dataset**: đẩy file (JSONL / JSON / CSV / TSV), chọn cột, lưu thành dataset. Dataset dùng cho build (train) và làm tập golden để eval.
2. **Prompt**: template (khung prompt có chỗ `{policy}`, `{input}` và phần output) và policy (đoạn luật). Mỗi lần sửa là một phiên bản mới, phiên bản cũ giữ nguyên.
3. **Build dataset**: ghép template + các cặp (policy, dataset) thành một phiên bản dữ liệu (`intent v1`), mỗi dòng thành một prompt hoàn chỉnh. Tuỳ chọn bỏ dòng trùng và chia train/val ngẫu nhiên (5–30%), hoặc thêm phần eval riêng. Có thêm dữ liệu thì mở build cũ, bấm **Tạo phiên bản mới** → `intent v2`. Build giữ bản sao dữ liệu và prompt đã dùng, nên train lại luôn ra đúng dữ liệu cũ.
4. **Finetune**: chọn build + model (model gốc, hoặc adapter đã thêm vào danh sách Model để train tiếp) + tham số (nhập trên form, hoặc chuyển sang **YAML** để sửa cả file `params.yaml`) → **Thêm vào hàng đợi**. Trong lúc train, model được chấm định kỳ trên phần val; trang job vẽ train loss / eval loss, các chỉ số khác (learning rate, grad norm, VRAM, accuracy, ngưỡng…) xem trên MLflow. **Dừng** là dừng ngay, không lưu checkpoint. Train xong giữ checkpoint có eval loss thấp nhất; **Thêm vào danh sách model** để bật và thử. Build có phần eval thì tự chấm sau khi train.
5. **Eval**: chọn model gốc → tick các adapter đã train từ nó → chọn build. Bảng xếp hạng: accuracy, macro-F1, ECE, và TP / FP / FN / TN, precision, recall, F1, coverage tại ngưỡng riêng của từng adapter; kèm F1 từng nhãn, cặp nhãn hay nhầm và câu sai.

- Hàng đợi GPU chung cho train và eval: mỗi lúc một việc, việc sau tự chạy. Tới lượt thì model đang chạy bị tắt; còn việc trong hàng đợi thì không bật được model.
- Dữ liệu: `data/finetune/{prompts,datasets,builds,jobs,evals}/<id>/`. Checkpoint: `~/model-platform/finetune/<id>` trong WSL.
- Platform gọi runtime theo [runtime protocol](docs/runtime-protocol.md), qua interface `RuntimeConnector` (`backend/finetune/connector.py`).

Thêm loại fine-tune mới (vd. chat SFT): lớp con của `FinetuneTask` (`backend/finetune/`, đăng ký trong `tasks.py`), lớp con của `Task` (`runtime/finetune/`, đăng ký trong `run.py`) và một file giao diện đăng ký trong `frontend/src/finetune/tasks.js`. Phần chung không cần sửa.

## MLflow: theo dõi train và model registry

- **Theo dõi:** mỗi job train là một run trên MLflow (experiment `finetune`), mở ngay khi job bắt đầu và cập nhật mỗi 5 giây trong lúc train; job kết thúc (xong, lỗi hoặc bị dừng) thì run đóng với trạng thái tương ứng. Run có toàn bộ chỉ số theo bước (`train_loss`, `train_lr`, `train_grad_norm`, `train_vram_gb`, `eval_loss`, `eval_acc`…, `final_*`), tham số và tag. Trang job trên platform chỉ vẽ loss / eval loss, có link sang run. MLflow tắt lúc job kết thúc thì bấm **Ghi lên MLflow** ở trang job để ghi bù.
- **Registry:** model train xong được **xuất bản** để `serving-platform` lấy về chạy.

- **MLflow server** (chạy bằng Docker) giữ metadata: tên model, phiên bản, alias, chỉ số, tag. Database SQLite nằm trong volume Docker `mlflow-db`.
- **File model** hiện lưu trên máy ở `data/mlflow/artifacts`. MLflow tự chuyển file (`--serve-artifacts`), nên platform và serving chỉ cần địa chỉ MLflow, không cần biết file nằm ở đâu.
- Sau này chuyển sang **Cloudflare R2** (để serving ở máy khác tải được): làm theo chú thích trong `docker-compose.yml`, không phải sửa code.

**Cài đặt (một lần):**

1. Cài và bật Docker Desktop.
2. Trong `.env` có `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`.
3. Ở thư mục project: `docker compose up -d` (lần đầu build image mất vài phút). Giao diện MLflow: http://127.0.0.1:5000. Tắt: `docker compose down`. Container tự bật lại khi Docker Desktop khởi động.

**Dùng:**

- Trang chi tiết job train đã xong → **Xuất bản** → đặt tên model (cùng tên thì thành phiên bản mới). Checkpoint được thêm vào run của job, kèm kết quả lần eval gần nhất có chấm job đó (chỉ số `golden_*`).
- Trang **Registry**: xem các phiên bản, gán alias **champion** (bản serving dùng) hoặc **challenger** (bản đang thử).

**Một phiên bản gồm:**

| Ở đâu | Nội dung |
| --- | --- |
| `model/` (file) | Checkpoint: LoRA, head, tokenizer, config, provenance |
| `platform/` (file) | `task_config.json` (nhãn, câu hỏi), `params.json`, `final.json` (ngưỡng tự động…), `eval_report.json` nếu đã eval |
| Tag (MLflow) | `prompt.template`, `prompt.policy` dạng `<tên>@v<N>`, model gốc, build, job, eval |
| Chỉ số (MLflow) | Lịch sử train của run (`train_*`, `eval_*`, `final_*`) và kết quả eval golden (`golden_acc`, `golden_macro_f1`, `golden_precision_at_threshold`…) |

Không có trong gói: model gốc (tải từ Hugging Face theo tag `base_model_repo`), dữ liệu train, và nội dung prompt (chỉ ghi phiên bản, nội dung lấy từ kho prompt).

Serving tải model theo alias:

```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
path = mlflow.artifacts.download_artifacts("models:/decider-intent@champion")
```

## Phát triển

Mở hai terminal:

```powershell
# Terminal 1: backend, tự nạp lại khi sửa file trong backend/
python platform_server.py --reload

# Terminal 2: giao diện, tự cập nhật khi sửa file trong frontend/src/
cd frontend
npm run dev
```

Rồi mở http://127.0.0.1:5173. Các request `/api` được chuyển tiếp tới backend ở cổng 8100.

### Kiểm tra code

```powershell
pip install -r requirements-dev.txt   # một lần: ruff, pytest

ruff check .                          # lint Python
pytest                                # test backend
cd frontend
npm run lint                          # lint giao diện
npm run build
```

- Test chạy trên một thư mục data tạm (biến `MODEL_PLATFORM_DATA`), không đụng dữ liệu thật, không cần WSL hay GPU.
- Đổi API (thêm, bớt, đổi đường dẫn) thì cập nhật `tests/routes.txt` cho khớp.

### CI

Mỗi lần push lên `main` hoặc mở pull request, GitHub Actions (`.github/workflows/ci.yml`) chạy đúng các lệnh kiểm tra ở trên trên máy Linux của GitHub. CI chỉ kiểm tra code, không chạy model hay GPU. Xem kết quả ở tab **Actions** của repo.

## Cấu trúc thư mục

```
platform_server.py      file khởi động (uvicorn cổng 8100, tuỳ chọn Vite dev server)
backend/                FastAPI
  config.py             đường dẫn WSL, cổng, VRAM: sửa ở đây khi đổi máy; nạp .env
  app.py                dựng app: middleware, lifespan, gộp router, phục vụ frontend/dist
  api/                  các API /api/...: system, models, playground, finetune
  core.py               hàm dùng chung: JSON, gọi WSL, log
  hf.py                 đọc thông tin model trên Hugging Face
  downloads.py          tải, tải lại, xoá model
  runner.py             bật/tắt server model trong WSL
  monitor.py            thông số GPU, trạng thái WSL
  presence.py           tự tắt khi không còn tab giao diện
  finetune/             prompt, dataset, build, job, eval, hàng đợi GPU, connector tới runtime, xuất bản
  model_registry.py     interface ModelRegistry (MLflow)
frontend/               React + Vite (src/pages, src/playground, src/finetune, src/components)
runtime/                chạy trong WSL: finetune/ (train, eval), image_server.py, tts_server.py
infra/mlflow/           Dockerfile của MLflow server (docker-compose.yml ở gốc)
tests/                  pytest
docs/                   runtime-protocol.md
examples/               script thử Strands Decider
data/                   dữ liệu chạy, riêng từng máy, không commit
```

## Gặp lỗi

| Hiện tượng | Cách xử lý |
| --- | --- |
| PowerShell báo `npm.ps1 cannot be loaded because running scripts is disabled` | Gõ `npm.cmd` thay cho `npm`, hoặc chạy một lần `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Trang báo "Chưa build giao diện" | Chạy `npm install` rồi `npm run build` trong thư mục `frontend` |
| Bấm shortcut mà không thấy gì mở ra | Xem `data/platform.log`, hoặc chạy `python platform_server.py --open` trong PowerShell để đọc lỗi |
| `ModuleNotFoundError: fastapi` (hoặc `dotenv`) | Chạy `pip install -r requirements.txt` bằng đúng bản Python mà shortcut dùng |
| Báo "cổng 8100 đang được dùng" nhưng không thấy gì chạy | Chạy `Get-NetTCPConnection -LocalPort 8100 -State Listen \| ForEach-Object { Stop-Process -Id $_.OwningProcess }` rồi khởi động lại |
| Trang Registry báo "Không kết nối được" | Bật Docker Desktop rồi chạy `docker compose up -d`; xem lỗi bằng `docker compose logs mlflow` |
| "Runtime … chưa được cài" | Vào trang Cài đặt → "Runtime trong WSL" để xem đường dẫn còn thiếu |
| Model gated (Llama, Gemma…) không tải được | Kiểm tra mục [Token Hugging Face](#token-hugging-face) và bấm đồng ý điều khoản trên trang model |
