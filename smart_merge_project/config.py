# config.py
from pathlib import Path

# ===== 路径（按你给的绝对路径）=====
RAW_DAILY_DIR = Path(r"D:\Pycharm\Pycharm date\pythonProject3\32. IEEE 5\smart_merge_project\data\raw_xlsx")
OUT_DIR = Path(r"D:\Pycharm\Pycharm date\pythonProject3\32. IEEE 5\smart_merge_project\data\output")

# ===== 数据过滤 =====
TARGET_MODEL = "ST12000NM0008"

# 你的 1–31 天就是文件名这31天
WINDOW_START = "2025-01-01"
WINDOW_END = "2025-01-31"

# ===== 负类抽样策略（按“每个xlsx抽30行 failure=0”）=====
NEG_SAMPLE_PER_FILE = 30
RANDOM_SEED = 42

# ===== 防泄漏与特征窗口 =====
LOOKBACK_DAYS = 30
EXCLUDE_FAILURE_DAY = True  # 正类 cutoff=first_failure_date-1day

LABEL_COL = "failed_in_window"

# ===== 特征列（沿用之前那套）=====
SMART_COLS = ["smart_5", "smart_187", "smart_188", "smart_197", "smart_198", "smart_199"]
SMART_TREND_COLS = ["smart_9", "smart_194", "smart_241", "smart_242"]

# ===== 输出文件名 =====
OUT_FILE = "disk_features_30d_noleak.csv"

# ===== 质量控制 =====
MIN_OBS_DAYS = 1