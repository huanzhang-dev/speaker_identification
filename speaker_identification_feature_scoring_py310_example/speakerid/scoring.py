from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .audio import LogMelFrontend, load_audio
from .model import SpeakerEncoder


def load_encoder_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = SpeakerEncoder(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    frontend = LogMelFrontend(**ckpt["feature_config"]).to(device)
    frontend.eval()
    return model, frontend, ckpt


@torch.no_grad()
def extract_embedding(
    model: SpeakerEncoder,
    frontend: LogMelFrontend,
    audio_path: str | Path,
    device: torch.device,
):
    waveform = load_audio(audio_path, frontend.sample_rate).to(device)
    feat = frontend(waveform)
    emb = model(feat)
    return emb[0].detach().cpu().numpy().astype(np.float32)


def l2_normalize_np(x: np.ndarray, axis: int = -1):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def build_gallery(speaker_ids: list[str], embeddings: np.ndarray):
    speaker_ids = np.asarray(speaker_ids, dtype=object)
    embeddings = l2_normalize_np(np.asarray(embeddings, dtype=np.float32))
    unique = sorted(set(speaker_ids.tolist()))
    prototypes = []
    for speaker in unique:
        e = embeddings[speaker_ids == speaker].mean(axis=0)
        prototypes.append(e)
    prototypes = l2_normalize_np(np.vstack(prototypes))
    return np.asarray(unique, dtype=object), prototypes.astype(np.float32)


def cosine_score_matrix(probe_embeddings, gallery_embeddings):
    """Higher is better."""
    p = l2_normalize_np(np.asarray(probe_embeddings, dtype=np.float32))
    g = l2_normalize_np(np.asarray(gallery_embeddings, dtype=np.float32))
    return p @ g.T


def euclidean_score_matrix(probe_embeddings, gallery_embeddings):
    """
    Negative Euclidean distance so that all project score matrices use
    the convention 'higher is better'.
    """
    p = np.asarray(probe_embeddings, dtype=np.float64)
    g = np.asarray(gallery_embeddings, dtype=np.float64)
    d2 = (
        np.sum(p * p, axis=1, keepdims=True)
        + np.sum(g * g, axis=1)[None, :]
        - 2.0 * p @ g.T
    )
    d = np.sqrt(np.maximum(d2, 0.0))
    return -d


def fit_mahalanobis_precision(
    background_embeddings: np.ndarray,
    regularization: float = 1e-3,
):
    """
    Estimate inverse covariance from background/training embeddings.

    For high-dimensional embeddings, regularization is essential.
    """
    x = np.asarray(background_embeddings, dtype=np.float64)
    cov = np.cov(x, rowvar=False)
    trace_scale = float(np.trace(cov) / max(cov.shape[0], 1))
    cov = cov + np.eye(cov.shape[0]) * max(
        regularization * trace_scale, 1e-8
    )
    return np.linalg.pinv(cov)


def mahalanobis_score_matrix(
    probe_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    precision: np.ndarray,
):
    """
    Negative Mahalanobis distance; higher is better.
    """
    p = np.asarray(probe_embeddings, dtype=np.float64)
    g = np.asarray(gallery_embeddings, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    scores = np.empty((len(p), len(g)), dtype=np.float64)
    for i, probe in enumerate(p):
        delta = g - probe[None, :]
        d2 = np.einsum("nd,dd,nd->n", delta, precision, delta)
        scores[i] = -np.sqrt(np.maximum(d2, 0.0))
    return scores


def score_matrix(
    probe_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    method: str = "cosine",
    precision: np.ndarray | None = None,
):
    method = method.lower()
    if method == "cosine":
        return cosine_score_matrix(probe_embeddings, gallery_embeddings)
    if method == "euclidean":
        return euclidean_score_matrix(probe_embeddings, gallery_embeddings)
    if method == "mahalanobis":
        if precision is None:
            raise ValueError("Mahalanobis scoring requires a precision matrix.")
        return mahalanobis_score_matrix(
            probe_embeddings, gallery_embeddings, precision
        )
    raise ValueError(f"Unknown score method: {method}")


def save_gallery(path: str | Path, speaker_ids, embeddings):
    np.savez_compressed(
        path,
        speaker_ids=np.asarray(speaker_ids, dtype=object),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )


def load_gallery(path: str | Path):
    data = np.load(path, allow_pickle=True)
    return data["speaker_ids"], data["embeddings"]
