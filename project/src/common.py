"""CR_Weight 公共工具：路径、配置、日志、确定性切分。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "project"
CONFIG_PATH = PROJECT / "config" / "config.yaml"
SQL_DIR = PROJECT / "sql"
DATA_DIR = PROJECT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"
FEATURE_DIR = DATA_DIR / "features"
RESULTS_DIR = PROJECT / "results"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"
DOCS_DIR = PROJECT / "docs"

# ---- 主模型标签（2026-10-08 起：以 ResNet-50 @308 三种子集成为主模型，见 docs/attention_ablation 结论）----
MAIN_TAG = os.environ.get("CR_MAIN_TAG", "resnet50_308_ensemble")

# 主模型逐种子标签前缀（checkpoint / metrics / 单种子预测文件名）。
MAIN_SEED_TAG = os.environ.get("CR_MAIN_SEED_TAG", "resnet50_308")

# 主模型在中文报告/图中的显示名（2026-10-08 起为 ResNet-50 @308 三种子集成）。
MAIN_LABEL = os.environ.get("CR_MAIN_LABEL", "ResNet-50 @308 3 种子集成（本项目）")


def main_pred_file() -> Path:
    """测试集主模型逐例预测文件。"""
    return PROCESSED_DIR / f"test_predictions_{MAIN_TAG}.csv"


def main_eval_file() -> Path:
    """测试集主模型评价结果。"""
    return TABLE_DIR / f"evaluation_{MAIN_TAG}.json"


def main_index_pred_file() -> Path:
    """队列 A 全部索引影像的主模型预测（供纵向/多模态分析使用）。"""
    return PROCESSED_DIR / f"preds_index_{MAIN_TAG}.csv"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, INTERIM_DIR, FEATURE_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str) -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    logfile = LOG_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("log file: %s", logfile)
    return logger


def stable_fraction(key: str, seed: int) -> float:
    """Deterministic [0,1) fraction from a string key (independent of pandas/numpy versions)."""
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, default=str)


def read_sql(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="utf-8")
