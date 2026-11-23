# Model Platform

Web chạy trên máy để tải, bật/tắt và thử các model AI chạy trong WSL (GPU RTX 3060 12 GB).

- Bảng điều khiển: http://127.0.0.1:8100
- Model đang chạy phục vụ ở http://127.0.0.1:8000 (mỗi lúc chỉ chạy một model)

## Cần có

| Thành phần | Ghi chú |
| --- | --- |
| Windows + WSL (distro `Ubuntu`) | Đã cài sẵn runtime: Strands Decider, vLLM, llama.cpp |
| Python 3.10 trở lên | Đã thử với 3.12 |
| Node.js 18 trở lên | Chỉ cần khi build hoặc sửa giao diện |
| Driver NVIDIA | Cần lệnh `nvidia-smi` để hiện thông số GPU |

## Cài đặt (làm một lần)

Mở PowerShell trong thư mục dự án:

```powershell
# 1. Thư viện Python cho backend
pip install -r requirements.txt

# 2. Build giao diện React
cd frontend
npm install
npm run build
cd ..
```

Sau bước 2 sẽ có thư mục `frontend/dist`, backend phục vụ giao diện từ thư mục này.

## Khởi động

Project chỉ chạy khi bạn khởi động nó, không có gì chạy ngầm hay tự bật cùng Windows.

**Cách 1: bấm đúp `Model Platform.lnk`.** Một cửa sổ console mở ra, server chạy trong đó và tự mở trình duyệt.

**Cách 2: dòng lệnh** (ở thư mục gốc dự án)

```powershell
python platform_server.py --open     # tự mở trình duyệt
python platform_server.py            # không mở trình duyệt

# Hoặc gọi thẳng uvicorn (cùng một ứng dụng FastAPI)
uvicorn backend.app:app --host 127.0.0.1 --port 8100
```

Nếu đã có một cửa sổ đang chạy thì lệnh mới chỉ báo "cổng 8100 đang được dùng" rồi thoát, không chạy server thứ hai.

Tài liệu API do FastAPI tự tạo: http://127.0.0.1:8100/docs

## Token Hugging Face

Cần cho model gated (Llama, Gemma…). Token không được lưu trong `data/` hay trên giao diện; nền tảng chỉ đọc biến môi trường `HF_TOKEN` của Windows.

**Đặt token (làm một lần).** Tạo token loại *Read* tại huggingface.co/settings/tokens, rồi chạy trong PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_xxxxxxxx", "User")
```

**Để có hiệu lực:**
1. Đóng và mở lại phiên đăng nhập Windows, hoặc khởi động lại Explorer trong Task Manager. Shortcut lấy biến từ Explorer nên cần bước này.
2. Tắt tiến trình `pythonw.exe` cũ trong Task Manager (nếu đang chạy), rồi mở lại shortcut. Server cũ không tự nhận token mới.

**Kiểm tra:** vào trang **Cài đặt** → mục *Token Hugging Face* phải hiện *Đã đặt hf_tk…xxxx* và tên tài khoản. Nếu hiện *Chưa đặt HF_TOKEN* thì làm lại hai bước trên.

**Đổi hoặc xóa token:** chạy lại lệnh trên với giá trị mới, hoặc thay `"hf_xxxxxxxx"` bằng `$null` để xóa. Sau đó làm lại bước 1 và 2.

## Tắt

**Đóng hết tab giao diện là nền tảng tự tắt** sau khoảng 5 giây. Khi tắt, server cũng tắt luôn model đang chạy trong WSL để trả VRAM. Tải lại trang (F5) không làm tắt server. Nếu cần tắt tay thì dùng Task Manager, kết thúc tiến trình `pythonw.exe`.

Trên giao diện còn có:
- **Tắt model** (trả VRAM, server vẫn chạy).
- **Tắt hẳn WSL** (tắt model và toàn bộ WSL, trả RAM cho Windows).

## Sửa code (chế độ phát triển)

Mở hai cửa sổ terminal:

```powershell
# Terminal 1: backend, tự nạp lại khi sửa file trong backend/
python platform_server.py --reload

# Terminal 2: giao diện, tự cập nhật khi sửa file trong frontend/src/
cd frontend
npm run dev
```

Rồi mở http://127.0.0.1:5173. Các request `/api` được chuyển tiếp tới backend ở cổng 8100.

Sửa xong giao diện thì chạy `npm run build` để bản ở cổng 8100 (bản shortcut mở) được cập nhật.

## Gặp lỗi

| Hiện tượng | Cách xử lý |
| --- | --- |
| PowerShell báo `npm.ps1 cannot be loaded because running scripts is disabled` | Gõ `npm.cmd` thay cho `npm` (ví dụ `npm.cmd run build`), hoặc chạy một lần `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Trang báo "Chưa build giao diện" | Chạy `npm install` rồi `npm run build` trong thư mục `frontend` |
| Bấm shortcut thì cửa sổ console đóng ngay | Mở PowerShell và chạy `python platform_server.py` để đọc lỗi |
| `ModuleNotFoundError: fastapi` | Chạy `pip install -r requirements.txt` bằng đúng bản Python mà shortcut dùng |
| Báo "cổng 8100 đang được dùng" nhưng không thấy cửa sổ nào | Chạy `Get-NetTCPConnection -LocalPort 8100 -State Listen \| ForEach-Object { Stop-Process -Id $_.OwningProcess }` rồi khởi động lại |
| "Runtime … chưa được cài" | Vào trang Cài đặt → "Runtime trong WSL" để xem đường dẫn còn thiếu |
| Model gated (Llama, Gemma…) không tải được | Kiểm tra mục [Token Hugging Face](#token-hugging-face) và bấm đồng ý điều khoản trên trang model |

## Cấu trúc thư mục

```
platform_server.py   file khởi động (uvicorn, cổng 8100)
backend/             FastAPI
  config.py          đường dẫn WSL, cổng, VRAM: sửa ở đây khi đổi máy
  app.py             các API /api/...
  hf.py              đọc thông tin model trên Hugging Face
  downloads.py       tải, tải lại, xoá model
  runner.py          bật/tắt server model trong WSL
  monitor.py         đọc thông số GPU, trạng thái WSL
frontend/            React + Vite (src/pages, src/playground, src/components)
data/                models.json (danh sách model), settings.json, platform.log (không chứa token)
examples/            script thử Strands Decider
runtime/             image_server.py: máy chủ sinh ảnh (diffusers), chạy trong venv ~/model-platform/runtimes/diffusers
```

## Dataset, Build dataset, Finetune, Eval

Bốn mục trên thanh bên, đi theo thứ tự:

1. **Dataset**: đẩy file (JSONL / JSON / CSV / TSV), chọn cột, lưu thành dataset. Dataset dùng cho build (train) và làm tập golden để eval.
2. **Build dataset**: ghép một hoặc nhiều dataset thành một phiên bản dữ liệu (`intent v1`), tuỳ chọn bỏ dòng trùng, chia train/val ngẫu nhiên (5–30%). Có thêm dữ liệu thì mở build cũ, bấm **Tạo phiên bản mới** (điền sẵn tên và dataset cũ), tick thêm dataset mới → `intent v2`. Build giữ bản sao dữ liệu nên train lại luôn ra đúng dữ liệu cũ.
3. **Finetune**: chọn build + model (model gốc, hoặc adapter đã thêm vào danh sách Model để train tiếp) + tham số → **Thêm vào hàng đợi**. Trong lúc train, model được chấm định kỳ trên phần val của build; dashboard có train loss / eval loss, learning rate, tốc độ, VRAM, grad norm, accuracy/F1/ECE trên val. **Dừng** là dừng ngay, không lưu checkpoint. Train xong giữ checkpoint có eval loss thấp nhất; **Thêm vào danh sách model** để bật và thử.
4. **Eval**: chọn model gốc → hiện các adapter đã train từ nó → tick adapter muốn so (không tick thì chỉ chấm model gốc) → chọn dataset golden. Bảng xếp hạng: accuracy, macro-F1, ECE, và TP / FP / FN / TN, precision, recall, F1, coverage tại ngưỡng riêng của từng adapter; kèm F1 từng nhãn của mọi model, cặp nhãn hay nhầm và câu sai.

- Hàng đợi GPU chung cho train và eval: mỗi lúc một việc, việc sau tự chạy. Tới lượt thì model đang chạy bị tắt; còn việc trong hàng đợi thì không bật được model và nền tảng không tự tắt khi đóng tab.
- Dữ liệu: `data/finetune/{datasets,builds,jobs,evals}/<id>/`. Checkpoint: `~/model-platform/finetune/<id>` trong WSL.

Thêm loại fine-tune mới (vd. chat SFT): lớp con của `FinetuneTask` (`backend/finetune/`, đăng ký trong `store.py`), lớp con của `Task` (`runtime/finetune/`, đăng ký trong `run.py`) và một file giao diện đăng ký trong `frontend/src/finetune/tasks.js`. Phần chung không cần sửa.
