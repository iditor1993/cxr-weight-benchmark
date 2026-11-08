"""根据填写完成的人工复核模板计算 Cohen's κ 与一致率。"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from common import save_json, setup_logger


def cohen_kappa(a: list[int], b: list[int]) -> float:
    a = pd.Series(a)
    b = pd.Series(b)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa1 = float((a == 1).mean())
    pb1 = float((b == 1).mean())
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--col1", required=True)
    ap.add_argument("--col2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    log = setup_logger("review_agreement")
    df = pd.read_csv(args.template)
    sub = df[[args.col1, args.col2]].dropna()
    if len(sub) == 0:
        log.warning("模板尚未填写任何评分；请先完成人工复核。")
        return
    a = sub[args.col1].astype(int).tolist()
    b = sub[args.col2].astype(int).tolist()
    out = {"n_rated": int(len(sub)), "n_total": int(len(df)),
           "agreement": float(sum(x == y for x, y in zip(a, b)) / len(a)),
           "cohen_kappa": cohen_kappa(a, b),
           "n_disagreements": int(sum(x != y for x, y in zip(a, b))),
           "n_ok_rater1": int(sum(a)), "n_ok_rater2": int(sum(b))}
    save_json(__import__("pathlib").Path(args.out), out)
    log.info(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
