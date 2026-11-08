"""等待训练队列结束后，补跑异方差（heteroscedastic）不确定性模型并评价其覆盖率。"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from common import PROJECT, TABLE_DIR, setup_logger

PY = sys.executable
SRC = PROJECT / "src"


def main() -> None:
    log = setup_logger("run_hetero")
    while not (TABLE_DIR / "metrics_rad_dino_224_s1.json").exists():
        log.info("waiting for training queue to finish")
        time.sleep(120)
    tag = "swin_t_308_hetero_s1"
    if not (TABLE_DIR / f"metrics_{tag}.json").exists():
        log.info("training heteroscedastic model")
        subprocess.run([PY, str(SRC / "train.py"), "--backbone", "swin_t", "--size", "308",
                        "--seed", "1", "--epochs", "30", "--patience", "6", "--head", "hetero",
                        "--tag", tag], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    ckpt = PROJECT / "data" / "features" / "checkpoints" / f"{tag}.pt"
    if ckpt.exists():
        subprocess.run([PY, str(SRC / "evaluate.py"), "--tag", "swin_hetero", "--backbone", "swin_t",
                        "--size", "308", "--head", "hetero", "--checkpoints", str(ckpt)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
        subprocess.run([PY, str(SRC / "uncertainty.py"), "--tag", "swin_hetero"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    log.info("hetero pipeline finished")


if __name__ == "__main__":
    main()
