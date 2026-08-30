from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from speakerid.metrics import (
    evaluate_gallery_scores,
    feature_geometry_metrics,
)
from speakerid.scoring import (
    build_gallery,
    extract_embedding,
    load_encoder_checkpoint,
    score_matrix,
)


def resolve_path(root, value):
    p = Path(str(value))
    return p if p.is_absolute() else root / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--enroll-manifest", type=Path, required=True)
    ap.add_argument("--probe-manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("."))
    ap.add_argument("--output-dir", type=Path, default=Path("eval_report"))
    ap.add_argument("--far-targets", nargs="+", type=float, default=[0.1, 0.01, 0.001])
    ap.add_argument("--score-method", choices=["cosine", "euclidean"], default="cosine")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    enroll_df = pd.read_csv(args.enroll_manifest)
    probe_df = pd.read_csv(args.probe_manifest)
    if not {"path", "speaker_id"}.issubset(enroll_df.columns):
        raise ValueError("Enrollment manifest needs path,speaker_id")
    if not {"path", "speaker_id"}.issubset(probe_df.columns):
        raise ValueError("Probe manifest needs path,speaker_id")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frontend, _ = load_encoder_checkpoint(args.checkpoint, device)

    enroll_labels, enroll_embeddings = [], []
    for _, row in enroll_df.iterrows():
        p = resolve_path(args.data_root, row["path"])
        enroll_embeddings.append(extract_embedding(model, frontend, p, device))
        enroll_labels.append(str(row["speaker_id"]))
    enroll_embeddings = np.vstack(enroll_embeddings)

    gallery_ids, gallery_embeddings = build_gallery(
        enroll_labels, enroll_embeddings
    )
    gallery_set = set(gallery_ids.tolist())

    probe_labels, probe_embeddings, is_known = [], [], []
    for _, row in probe_df.iterrows():
        p = resolve_path(args.data_root, row["path"])
        probe_embeddings.append(extract_embedding(model, frontend, p, device))
        sid = str(row["speaker_id"])
        probe_labels.append(sid)
        if "is_known" in probe_df.columns:
            value = row["is_known"]
            if isinstance(value, str):
                known_value = value.strip().lower() in {"1", "true", "yes", "y"}
            else:
                known_value = bool(value)
            is_known.append(known_value)
        else:
            is_known.append(sid in gallery_set)

    probe_embeddings = np.vstack(probe_embeddings)
    scores = score_matrix(probe_embeddings, gallery_embeddings, method=args.score_method)

    score_metrics = evaluate_gallery_scores(
        scores,
        probe_labels,
        gallery_ids.tolist(),
        is_known,
        far_targets=args.far_targets,
    )

    # Geometry uses enrollment + known probes only.
    geo_embeddings = [enroll_embeddings]
    geo_labels = list(enroll_labels)
    known_probe_emb = []
    known_probe_labels = []
    for emb, sid, known in zip(probe_embeddings, probe_labels, is_known):
        if known:
            known_probe_emb.append(emb)
            known_probe_labels.append(sid)
    if known_probe_emb:
        geo_embeddings.append(np.vstack(known_probe_emb))
        geo_labels.extend(known_probe_labels)
    geometry = feature_geometry_metrics(
        np.vstack(geo_embeddings),
        geo_labels,
    )

    report = {
        "score_metrics": score_metrics,
        "feature_geometry": geometry,
    }
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # CSV with top result and margin per probe.
    order = np.argsort(scores, axis=1)[:, ::-1]
    rows = []
    for i in range(len(probe_df)):
        best = order[i, 0]
        second = order[i, 1] if scores.shape[1] > 1 else best
        rows.append(
            {
                "path": probe_df.iloc[i]["path"],
                "speaker_id": probe_labels[i],
                "is_known": bool(is_known[i]),
                "predicted_speaker": str(gallery_ids[best]),
                "top1_score": float(scores[i, best]),
                "top2_score": float(scores[i, second]),
                "top1_top2_margin": float(scores[i, best] - scores[i, second]),
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "probe_scores.csv", index=False)

    # Genuine / impostor histogram.
    gallery_index = {s: i for i, s in enumerate(gallery_ids.tolist())}
    genuine, impostor = [], []
    for i, sid in enumerate(probe_labels):
        if is_known[i] and sid in gallery_index:
            gi = gallery_index[sid]
            genuine.append(scores[i, gi])
            mask = np.arange(scores.shape[1]) != gi
            impostor.extend(scores[i, mask].tolist())
        elif not is_known[i]:
            impostor.extend(scores[i].tolist())

    if genuine and impostor:
        plt.figure(figsize=(7, 4.2))
        plt.hist(impostor, bins=60, density=True, alpha=0.55, label="Impostor")
        plt.hist(genuine, bins=40, density=True, alpha=0.55, label="Genuine")
        plt.xlabel(f"{args.score_method.capitalize()} score (higher is better)")
        plt.ylabel("Density")
        plt.title("Genuine vs impostor score distributions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "genuine_impostor_scores.png", dpi=170)
        plt.close()

    print(json.dumps(report, indent=2))
    print("Saved:", args.output_dir / "metrics.json")
    print("Saved:", args.output_dir / "probe_scores.csv")


if __name__ == "__main__":
    main()
