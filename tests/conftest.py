"""Mọi test chạy trên một thư mục data tạm, đặt trước khi import backend (config đọc biến lúc import)."""

import os
import tempfile

os.environ["MODEL_PLATFORM_DATA"] = tempfile.mkdtemp(prefix="model-platform-test-")
