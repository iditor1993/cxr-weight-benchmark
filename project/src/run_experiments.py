"""实验队列：按顺序训练基线/主模型，并输出队列日志。"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from common import PROJECT, TABLE_DIR, setup_logger

PY = sys.executable
TRAIN = str(PROJECT / "src" / "train.py")

EXPERIMENTS = [
    # (backbone, size, seed, epochs, patience, split, tag)
    ("resnet50", 224, 1, 30, 6, "random", "resnet50_224_s1"),
    ("swin_t", 308, 1, 30, 6, "random", "swin_t_308_s1"),
    ("swin_t", 308, 2, 30, 6, "random", "swin_t_308_s2"),
    ("swin_t", 308, 3, 30, 6, "random", "swin_t_308_s3"),
    ("swin_t", 224, 1, 30, 6, "random", "swin_t_224_s1"),
    ("swin_t", 308, 1, 30, 6, "temporal", "swin_t_308_temporal_s1"),
    ("rad_dino", 224, 1, 25, 6, "random", "rad_dino_224_s1"),
]


def main() -> None:
    log = setup_logger("run_experiments")
    for backbone, size, seed, epochs, patience, split, tag in EXPERIMENTS:
        metrics_file = TABLE_DIR / f"metrics_{tag}.json"
        if metrics_file.exists():
            log.info("skip %s (exists)", tag)
            continue
        cmd = [PY, TRAIN, "--backbone", backbone, "--size", str(size), "--split", split,
               "--seed", str(seed), "--epochs", str(epochs),
               "--patience", str(patience), "--tag", tag]
        log.info("START %s", " ".join(cmd))
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (Path(PROJECT / "results" / "logs") / f"queue_{tag}.log").write_text(
            (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or ""), encoding="utf-8")
        log.info("END %s rc=%d elapsed=%.1f min", tag, proc.returncode, (time.time() - t0) / 60)
    log.info("queue finished")


if __name__ == "__main__":
    main()
