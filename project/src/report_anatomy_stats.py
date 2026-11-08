"""打印全量解剖遮挡的配对统计（均值、95% CI、p 值）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import TABLE_DIR

d = json.loads((TABLE_DIR / "anatomical_occlusion.json").read_text(encoding="utf-8"))
print("full-test occlusion n_images =", d["n_images"])
for k, v in d["regions"].items():
    a = np.array(v.get("per_image_delta_err", []), dtype=float)
    b = np.array(v.get("per_image_delta_err_shift", []), dtype=float)
    if len(a) == 0:
        continue
    if len(b) == len(a):
        diff = a - b
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"{k:18s} n={len(a):4d} region={a.mean():+.2f} control={b.mean():+.2f} "
              f"paired={diff.mean():+.2f} CI95=[{diff.mean()-1.96*se:+.2f},{diff.mean()+1.96*se:+.2f}] "
              f"p={v.get('wilcoxon_region_vs_shift_p'):.2e} frac_worse={v.get('frac_region_worse'):.2f} "
              f"area={v['area_mean']:.3f}")
    else:
        se = a.std(ddof=1) / np.sqrt(len(a))
        print(f"{k:18s} n={len(a):4d} region={a.mean():+.2f} "
              f"CI95=[{a.mean()-1.96*se:+.2f},{a.mean()+1.96*se:+.2f}] area={v['area_mean']:.3f}")
