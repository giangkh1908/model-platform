"""Điểm vào chạy trong WSL:
    python run.py train <task> <job_dir>
    python run.py eval <task> <eval_dir>
Thêm loại fine-tune mới: viết một lớp con của common.Task rồi đăng ký vào TASKS."""

from common import main
from decider import DeciderTask

TASKS = {"decider": DeciderTask}

if __name__ == "__main__":
    main(TASKS)
