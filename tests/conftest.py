"""Đặt môi trường cho test trước khi import backend (config đọc biến lúc import):
- data trên thư mục tạm, không đụng dữ liệu thật;
- tắt Langfuse và MLflow dù .env của máy có key (đặt rỗng thì load_dotenv không ghi đè), để test không ghi lên dịch vụ thật.
"""

import os
import tempfile

os.environ["MODEL_PLATFORM_DATA"] = tempfile.mkdtemp(prefix="model-platform-test-")
for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "MLFLOW_TRACKING_URI"):
    os.environ[name] = ""
