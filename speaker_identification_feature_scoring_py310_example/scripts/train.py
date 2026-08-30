from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from speakerid.dataset import SpeakerBalancedBatchSampler, SpeakerDataset
from speakerid.losses import AAMSoftmax
from speakerid.model import SpeakerEncoder


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_mapping(train_manifest):
    df = pd.read_csv(train_manifest)
    speakers = sorted(df["speaker_id"].astype(str).unique().tolist())
    return {s: i for i, s in enumerate(speakers)}


def run_epoch(model, head, loader, device, optimizer=None, grad_clip=5.0):
    training = optimizer is not None
    model.train(training)
    head.train(training)

    losses = []
    correct = 0
    total = 0

    for feat, labels in loader:
        feat = feat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        emb = model(feat)
        logits = head(emb, labels)
        loss = nn.functional.cross_entropy(logits, labels)

        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()),
                grad_clip,
            )
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        pred = logits.detach().argmax(dim=1)
        correct += int((pred == labels).sum().cpu())
        total += len(labels)

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "classification_accuracy": correct / max(total, 1),
    }


def save_checkpoint(path, model, head, config, speaker_to_index, metrics):
    torch.save(
        {
            "model_state": model.state_dict(),
            "aam_state": head.state_dict(),
            "model_config": config["model"],
            "feature_config": config["features"],
            "speaker_to_index": speaker_to_index,
            "metrics": metrics,
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config.json")
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--dev-manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("."))
    ap.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "exp01")
    ap.add_argument("--pretrained-encoder", type=Path, default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    seed_all(config["training"]["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    speaker_to_index = build_mapping(args.train_manifest)
    print("Training speakers:", len(speaker_to_index))

    train_ds = SpeakerDataset(
        args.train_manifest,
        args.data_root,
        speaker_to_index,
        config["features"],
        segment_seconds=config["training"]["segment_seconds"],
        training=True,
        noise_probability=config["training"]["noise_probability"],
        gain_probability=config["training"]["gain_probability"],
        freq_mask=config["training"]["freq_mask"],
        time_mask=config["training"]["time_mask"],
    )
    dev_ds = SpeakerDataset(
        args.dev_manifest,
        args.data_root,
        speaker_to_index,
        config["features"],
        segment_seconds=config["training"]["segment_seconds"],
        training=False,
        noise_probability=0.0,
        gain_probability=0.0,
        freq_mask=0,
        time_mask=0,
    )

    sampler = SpeakerBalancedBatchSampler(
        train_ds.labels,
        speakers_per_batch=config["training"]["speakers_per_batch"],
        utterances_per_speaker=config["training"]["utterances_per_speaker"],
        seed=config["training"]["seed"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cpu")
    print("Device:", device)

    model = SpeakerEncoder(**config["model"]).to(device)
    head = AAMSoftmax(
        embedding_dim=config["model"]["embedding_dim"],
        num_classes=len(speaker_to_index),
        margin=config["loss"]["margin"],
        scale=config["loss"]["scale"],
    ).to(device)

    if args.pretrained_encoder is not None:
        payload = torch.load(
            args.pretrained_encoder, map_location="cpu"
        )
        state = payload.get("model_state", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded pretrained encoder:", args.pretrained_encoder)
        print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "lr": config["training"]["encoder_lr"],
            },
            {
                "params": head.parameters(),
                "lr": config["training"]["head_lr"],
            },
        ],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )

    best_loss = float("inf")
    history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        sampler.set_epoch(epoch)
        train_m = run_epoch(
            model, head, train_loader, device, optimizer,
            config["training"]["grad_clip"]
        )
        dev_m = run_epoch(model, head, dev_loader, device)
        scheduler.step()

        row = {"epoch": epoch, "train": train_m, "dev": dev_m}
        history.append(row)
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_m['loss']:.4f} acc {train_m['classification_accuracy']:.4f} | "
            f"dev loss {dev_m['loss']:.4f} acc {dev_m['classification_accuracy']:.4f}"
        )

        save_checkpoint(
            args.output_dir / "last.pt",
            model, head, config, speaker_to_index, dev_m
        )

        if dev_m["loss"] < best_loss:
            best_loss = dev_m["loss"]
            save_checkpoint(
                args.output_dir / "best_encoder.pt",
                model, head, config, speaker_to_index, dev_m
            )
            print("  -> saved best_encoder.pt")

    with open(args.output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(args.output_dir / "speaker_mapping.json", "w", encoding="utf-8") as f:
        json.dump(speaker_to_index, f, indent=2)

    print("Done. Best checkpoint:", args.output_dir / "best_encoder.pt")


if __name__ == "__main__":
    main()
