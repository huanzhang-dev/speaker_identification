from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def threshold_for_far(scores: np.ndarray, target_far: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = len(scores)
    if n == 0:
        raise ValueError("No unknown-speaker scores were supplied.")

    allowed = int(np.floor(target_far * n))
    if allowed <= 0:
        return float(np.nextafter(scores[0], np.inf))
    if allowed >= n:
        return float(np.nextafter(scores[-1], -np.inf))
    return float((scores[allowed - 1] + scores[allowed]) / 2.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate an open-set speaker-identification threshold."
    )
    ap.add_argument(
        "--probe-scores",
        type=Path,
        required=True,
        help="probe_scores.csv produced by scripts/evaluate.py",
    )
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--output", type=Path, default=Path("threshold.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.probe_scores)
    if "is_known" not in df.columns or "top1_score" not in df.columns:
        raise ValueError(
            "CSV must contain columns `is_known` and `top1_score`."
        )

    unknown = df.loc[~df["is_known"].astype(bool), "top1_score"].to_numpy()
    threshold = threshold_for_far(unknown, args.target_far)
    empirical_far = float(np.mean(unknown >= threshold))

    result = {
        "target_far": float(args.target_far),
        "threshold": float(threshold),
        "empirical_far": empirical_far,
        "unknown_probe_count": int(len(unknown)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
