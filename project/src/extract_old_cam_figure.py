"""从 git 历史提取修复前的错误 CAM 图，仅作为修复记录的 before 证据。"""

from __future__ import annotations

import subprocess
from pathlib import Path

from common import DOCS_DIR, setup_logger

REV = "3d8d572"
SRC = "project/results/figures/fig_attention_swin_ensemble_0.png"
OUT = DOCS_DIR / "cam_evidence" / "before_fix_attention_example.png"


def main() -> None:
    log = setup_logger("extract_old_cam_figure")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = subprocess.check_output(["git", "show", f"{REV}:{SRC}"],
                                   cwd=str(Path(__file__).resolve().parents[2]))
    OUT.write_bytes(blob)
    log.info("before-fix figure (%d bytes) -> %s", len(blob), OUT)


if __name__ == "__main__":
    main()
