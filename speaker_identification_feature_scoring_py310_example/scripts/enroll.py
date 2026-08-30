from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from speakerid.scoring import (
    build_gallery,
    extract_embedding,
    load_encoder_checkpoint,
    save_gallery,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, default=Path("gallery.npz"))
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if not {"path", "speaker_id"}.issubset(df.columns):
        raise ValueError("Enrollment manifest needs path,speaker_id")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frontend, _ = load_encoder_checkpoint(args.checkpoint, device)

    labels = []
    embeddings = []
    for _, row in df.iterrows():
        p = Path(str(row["path"]))
        if not p.is_absolute():
            p = args.data_root / p
        emb = extract_embedding(model, frontend, p, device)
        labels.append(str(row["speaker_id"]))
        embeddings.append(emb)

    gallery_ids, gallery_embeddings = build_gallery(
        labels, np.vstack(embeddings)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_gallery(args.output, gallery_ids, gallery_embeddings)
    print(f"Saved {len(gallery_ids)} enrolled speakers -> {args.output}")


if __name__ == "__main__":
    main()
