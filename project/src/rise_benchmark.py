"""RISE 掩膜数量与吞吐基准：不同 batch size 下 Swin-T@308 的前向速度与显存峰值。"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF

from common import FEATURE_DIR, TABLE_DIR, save_json, setup_logger
from dataset import WeightCXR
from models import build_model, normalization_for


def main() -> None:
    log = setup_logger("rise_benchmark")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    props = torch.cuda.get_device_properties(0)
    log.info("GPU: %s, total VRAM %.1f GB", props.name, props.total_memory / 1e9)

    ckpt = torch.load(FEATURE_DIR / "checkpoints" / "swin_t_308_s1.pt",
                      map_location=device, weights_only=False)
    model = build_model("swin_t", img_size=308).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean, std = normalization_for("swin_t")
    ds = WeightCXR("test", split_kind="random", size=308, train=False, mean=mean, std=std)
    x = ds[0]["image"].unsqueeze(0).to(device)
    x_blur = TF.gaussian_blur(x, kernel_size=61, sigma=20.0)
    side, cell = 308, 16

    results = {"gpu": props.name, "total_vram_gb": props.total_memory / 1e9,
               "input_size": side, "cell": cell, "batches": []}
    best = None
    for bs in (8, 32, 64, 128, 192, 256, 384):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # 预热
            low = (torch.rand(bs, 1, cell, cell, device=device) > 0.5).float()
            for _ in range(2):
                m = torch.nn.functional.interpolate(low, size=(side, side), mode="bilinear",
                                                    align_corners=False)
                m = (m > 0.5).float()
                xb = x.repeat(bs, 1, 1, 1) * m + x_blur.repeat(bs, 1, 1, 1) * (1 - m)
                with torch.no_grad():
                    model(xb)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            reps = 5
            for _ in range(reps):
                low = (torch.rand(bs, 1, cell, cell, device=device) > 0.5).float()
                m = torch.nn.functional.interpolate(low, size=(side, side), mode="bilinear",
                                                    align_corners=False)
                m = (m > 0.5).float()
                xb = x.repeat(bs, 1, 1, 1) * m + x_blur.repeat(bs, 1, 1, 1) * (1 - m)
                with torch.no_grad():
                    model(xb)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / reps
            peak = torch.cuda.max_memory_allocated() / 1e9
            rec = {"batch_size": bs, "seconds_per_batch": dt, "ms_per_mask": dt / bs * 1000,
                   "masks_per_second": bs / dt, "peak_vram_gb": peak}
            results["batches"].append(rec)
            log.info("bs=%3d: %.3f s/batch | %.1f ms/mask | %.0f masks/s | peak %.2f GB",
                     bs, dt, dt / bs * 1000, bs / dt, peak)
            if best is None or rec["masks_per_second"] > best["masks_per_second"]:
                best = rec
        except torch.cuda.OutOfMemoryError:
            log.warning("bs=%d OOM", bs)
            results["batches"].append({"batch_size": bs, "error": "OOM"})
            torch.cuda.empty_cache()

    results["best"] = best
    mps = best["masks_per_second"]
    scenarios = {}
    for masks in (500, 1000, 2000, 4000, 8000):
        for n_img in (30, 60, 200):
            sec = masks * n_img / mps
            scenarios[f"{masks}_masks_x{n_img}_images"] = {
                "seconds": sec, "minutes": sec / 60, "hours": sec / 3600}
    results["scenarios"] = scenarios
    save_json(TABLE_DIR / "rise_benchmark.json", results)
    log.info("best batch=%d, %.0f masks/s", best["batch_size"], mps)
    for masks in (2000, 4000, 8000):
        for n in (30, 60, 200):
            k = f"{masks}_masks_x{n}_images"
            log.info("%d masks x %d images -> %.1f min", masks, n, scenarios[k]["minutes"])


if __name__ == "__main__":
    main()
