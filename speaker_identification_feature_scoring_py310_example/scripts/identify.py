from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from speakerid.scoring import (
    extract_embedding,
    load_encoder_checkpoint,
    load_gallery,
    score_matrix,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--gallery", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--score-method", choices=["cosine", "euclidean"], default="cosine")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frontend, _ = load_encoder_checkpoint(args.checkpoint, device)
    gallery_ids, gallery_embeddings = load_gallery(args.gallery)

    emb = extract_embedding(model, frontend, args.audio, device)
    scores = score_matrix(emb[None, :], gallery_embeddings, method=args.score_method)[0]
    order = np.argsort(scores)[::-1]
    k = min(args.topk, len(order))

    print("Top matches:")
    for rank, idx in enumerate(order[:k], start=1):
        print(f"{rank:2d}. {gallery_ids[idx]}  score={scores[idx]:.6f}")

    best = int(order[0])
    margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else float("nan")
    print(f"Top1-Top2 margin: {margin:.6f}")

    if scores[best] < args.threshold:
        print(f"Decision: UNKNOWN (best score {scores[best]:.6f} < {args.threshold:.6f})")
    else:
        print(f"Decision: {gallery_ids[best]} (score={scores[best]:.6f})")


if __name__ == "__main__":
    main()
