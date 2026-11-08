"""从可用的 GitHub 镜像下载 chestx_anatomy 分割权重到 torchxrayvision 缓存目录。"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import requests
import torchxrayvision as xrv

from common import setup_logger

log = setup_logger("download_cxas_weights")
URL_PATH = ("mlmed/torchxrayvision/releases/download/v1/"
            "baseline_models_cxas_UNet_ResNet50_default_repack.pth")
MIRRORS = ["https://ghproxy.net/https://github.com/", "https://gh-proxy.com/https://github.com/",
           "https://ghfast.top/https://github.com/"]
FILENAME = os.path.basename(URL_PATH)
DEST = Path(xrv.utils.get_cache_dir()) / FILENAME
EXPECTED = 293_894_593


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size == EXPECTED:
        log.info("already downloaded: %s", DEST)
        return
    tmp = DEST.with_suffix(".part")
    for mirror in MIRRORS:
        url = mirror + URL_PATH
        log.info("trying %s", mirror)
        try:
            t0 = time.time()
            with requests.get(url, stream=True, timeout=30, allow_redirects=True) as r:
                if r.status_code != 200:
                    log.warning("HTTP %s from %s", r.status_code, mirror)
                    continue
                total = int(r.headers.get("content-length", 0))
                done = 0
                last = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        fh.write(chunk)
                        done += len(chunk)
                        if done - last > 20 * 1024 * 1024:
                            last = done
                            speed = done / max(time.time() - t0, 1e-6) / 1e6
                            log.info("  %.0f/%.0f MB (%.1f MB/s)", done / 1e6, total / 1e6, speed)
            if tmp.stat().st_size == EXPECTED:
                shutil.move(str(tmp), str(DEST))
                log.info("downloaded OK -> %s (%.0f MB, %.1f min)",
                         DEST, DEST.stat().st_size / 1e6, (time.time() - t0) / 60)
                return
            log.warning("size mismatch: %d != %d", tmp.stat().st_size, EXPECTED)
            tmp.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("mirror failed: %s: %s", type(exc).__name__, str(exc)[:120])
            tmp.unlink(missing_ok=True)
    raise RuntimeError("all mirrors failed")


if __name__ == "__main__":
    main()
