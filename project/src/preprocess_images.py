"""Stage 2b：DICOM → 320/224 px uint8 npy 缓存 + 自动质控。

用法：
  python project/src/preprocess_images.py --set index --workers 8
  python project/src/preprocess_images.py --set all   --workers 8
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from common import FEATURE_DIR, PROCESSED_DIR, load_config, setup_logger

LOGGER = None


def _window_to_uint8(arr: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (out * 255.0).round().astype(np.uint8)


def _pad_to_square(img: Image.Image, fill: int = 0) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), color=fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def process_one(row: dict, size: int, out_dir: Path, low_pct: float, high_pct: float) -> dict:
    dicom_id = row["dicom_id"]
    out_path = out_dir / f"{dicom_id}.npy"
    qc = {
        "dicom_id": dicom_id,
        "ok": False,
        "error": "",
        "rows": None,
        "cols": None,
        "kvp": None,
        "exposure_index": None,
        "imager_pixel_spacing": None,
        "mean_intensity": None,
        "frac_saturated": None,
    }
    if out_path.exists():
        qc["ok"] = True
        qc["error"] = "cached"
        return qc
    try:
        with zipfile.ZipFile(row["zip_path"]) as zf:
            data = zf.read(row["inner_path"])
        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array
        qc["rows"], qc["cols"] = int(ds.Rows), int(ds.Columns)
        qc["kvp"] = float(ds.KVP) if "KVP" in ds else None
        qc["exposure_index"] = float(ds.ExposureIndex) if "ExposureIndex" in ds else None
        spacing = ds.get("ImagerPixelSpacing") or ds.get("PixelSpacing")
        qc["imager_pixel_spacing"] = float(spacing[0]) if spacing else None
        if arr.ndim == 3:
            arr = arr[..., 0]
        qc["mean_intensity"] = float(arr.mean())
        qc["frac_saturated"] = float((arr >= arr.max()).mean())
        img = Image.fromarray(_window_to_uint8(arr, low_pct, high_pct), mode="L")
        img = _pad_to_square(img)
        img = img.resize((size, size), Image.BILINEAR)
        np.save(out_path, np.asarray(img, dtype=np.uint8))
        qc["ok"] = True
    except Exception as exc:  # noqa: BLE001
        qc["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return qc


def run(set_name: str, workers: int, size: int, limit: int | None = None) -> None:
    global LOGGER
    LOGGER = setup_logger(f"preprocess_{set_name}")
    cfg = load_config()
    manifest = PROCESSED_DIR / f"image_manifest_{set_name}.csv"
    df = pd.read_csv(manifest)
    if limit:
        df = df.head(limit)
    out_dir = FEATURE_DIR / f"{set_name}_{size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("processing %d images -> %s (workers=%d, size=%d)", len(df), out_dir, workers, size)

    # 按 zip_path 排序，减少同一 zip 的重复打开
    df = df.sort_values(["zip_path", "inner_path"]).reset_index(drop=True)
    rows = df.to_dict("records")
    qc_rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one, r, size, out_dir, cfg["image"]["percentile_low"],
                             cfg["image"]["percentile_high"]) for r in rows]
        for fut in as_completed(futures):
            qc_rows.append(fut.result())
            done += 1
            if done % 500 == 0:
                LOGGER.info("progress %d/%d", done, len(rows))

    qc_df = pd.DataFrame(qc_rows)
    qc_path = PROCESSED_DIR / f"qc_images_{set_name}_{size}.csv"
    qc_df.to_csv(qc_path, index=False)
    ok = int(qc_df["ok"].sum())
    LOGGER.info("done: ok=%d/%d (%.1f%%), qc -> %s", ok, len(qc_df), 100.0 * ok / max(len(qc_df), 1), qc_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", default="index",
                    choices=["index", "all", "cohort_b", "nrr"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.set_name, args.workers, args.size, args.limit)


if __name__ == "__main__":
    main()
