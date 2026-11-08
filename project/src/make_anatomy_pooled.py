"""把 308×308 分割掩膜池化为 20×20 平均图（逐区域加载，控制内存）。"""

from __future__ import annotations

import numpy as np

from common import PROCESSED_DIR, TABLE_DIR, save_json, setup_logger

REGIONS = ["lungs", "cardiomediastinum", "subdiaphragmatic", "liver"]


def main() -> None:
    log = setup_logger("make_anatomy_pooled")
    src = PROCESSED_DIR / "anatomy_regions_test1303.npz"
    npz = np.load(src, allow_pickle=True)
    pooled = {}
    for k in REGIONS:
        arr = npz[k]                                   # (N,308,308) uint8
        n = arr.shape[0]
        p = arr[:, :300, :300].reshape(n, 20, 15, 20, 15).mean(axis=(2, 4))
        pooled[k] = p.mean(0).astype(np.float32)
        log.info("%s: n=%d pooled mean=%.3f", k, n, float(pooled[k].mean()))
    np.savez_compressed(PROCESSED_DIR / "anatomy_pooled_1303.npz", **pooled)
    log.info("saved -> %s", PROCESSED_DIR / "anatomy_pooled_1303.npz")


if __name__ == "__main__":
    main()
