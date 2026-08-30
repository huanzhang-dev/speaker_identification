from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)


def _safe_mean(x):
    return float(np.mean(x)) if len(x) else float("nan")


def _safe_std(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def dprime(genuine: np.ndarray, impostor: np.ndarray):
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    if len(genuine) < 2 or len(impostor) < 2:
        return float("nan")
    num = abs(genuine.mean() - impostor.mean())
    den = math.sqrt(0.5 * (genuine.var(ddof=1) + impostor.var(ddof=1)))
    return float(num / den) if den > 0 else float("inf")


def eer_from_scores(genuine: np.ndarray, impostor: np.ndarray):
    if len(genuine) == 0 or len(impostor) == 0:
        return float("nan"), float("nan")
    y = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    s = np.concatenate([genuine, impostor])
    fpr, tpr, thresholds = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    eer = 0.5 * (fpr[i] + fnr[i])
    return float(eer), float(thresholds[i])


def feature_geometry_metrics(
    embeddings: np.ndarray,
    labels: list[str],
    silhouette_max_samples: int = 3000,
    seed: int = 42,
):
    """
    Geometry metrics on L2-normalized embeddings.

    within_cosine_distance: lower is better
    between_centroid_cosine_distance: higher is better
    separation_ratio: higher is better
    fisher_trace_ratio: higher is better
    silhouette_cosine: higher is better
    """
    x = np.asarray(embeddings, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    labels = np.asarray(labels, dtype=object)

    speakers = sorted(set(labels.tolist()))
    centroids = {}
    within = []

    for spk in speakers:
        e = x[labels == spk]
        c_raw = e.mean(axis=0)
        c = c_raw / max(np.linalg.norm(c_raw), 1e-12)
        centroids[spk] = c
        within.extend((1.0 - e @ c).tolist())

    centroid_matrix = np.vstack([centroids[s] for s in speakers])
    between = []
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            between.append(1.0 - float(centroid_matrix[i] @ centroid_matrix[j]))

    within_mean = _safe_mean(within)
    between_mean = _safe_mean(between)
    separation_ratio = (
        float(between_mean / max(within_mean, 1e-12))
        if not math.isnan(within_mean) and not math.isnan(between_mean)
        else float("nan")
    )

    # Fisher trace ratio without constructing D x D scatter matrices.
    global_mean = x.mean(axis=0)
    sw = 0.0
    sb = 0.0
    for spk in speakers:
        e = x[labels == spk]
        c = e.mean(axis=0)
        sw += float(np.sum((e - c) ** 2))
        sb += len(e) * float(np.sum((c - global_mean) ** 2))
    fisher = sb / max(sw, 1e-12)

    # Silhouette: keep only classes with >=2 samples.
    counts = Counter(labels.tolist())
    keep = np.array([counts[v] >= 2 for v in labels])
    xs = x[keep]
    ys = labels[keep]
    sil = float("nan")
    if len(set(ys.tolist())) >= 2 and len(xs) > len(set(ys.tolist())):
        rng = np.random.default_rng(seed)
        if len(xs) > silhouette_max_samples:
            idx = rng.choice(len(xs), size=silhouette_max_samples, replace=False)
            xs, ys = xs[idx], ys[idx]
        try:
            sil = float(silhouette_score(xs, ys, metric="cosine"))
        except ValueError:
            sil = float("nan")

    return {
        "within_speaker_cosine_distance_mean": within_mean,
        "between_speaker_centroid_cosine_distance_mean": between_mean,
        "between_within_separation_ratio": separation_ratio,
        "fisher_trace_ratio": float(fisher),
        "silhouette_cosine": sil,
    }


def _threshold_for_target_far(unknown_max_scores: np.ndarray, target_far: float):
    """
    Empirical threshold chosen so FAR <= target_far.
    """
    s = np.sort(np.asarray(unknown_max_scores, dtype=float))[::-1]
    n = len(s)
    if n == 0:
        return float("nan")
    allowed = int(math.floor(target_far * n))
    if allowed <= 0:
        return float(np.nextafter(s[0], np.inf))
    if allowed >= n:
        return float(np.nextafter(s[-1], -np.inf))
    return float((s[allowed - 1] + s[allowed]) / 2.0)


def evaluate_gallery_scores(
    score_matrix: np.ndarray,
    probe_speaker_ids: list[str],
    gallery_speaker_ids: list[str],
    is_known: list[bool] | np.ndarray,
    far_targets=(0.1, 0.01, 0.001),
    max_impostor_scores: int = 200000,
    seed: int = 42,
):
    scores = np.asarray(score_matrix, dtype=float)
    probe_ids = np.asarray(probe_speaker_ids, dtype=object)
    gallery_ids = np.asarray(gallery_speaker_ids, dtype=object)
    known = np.asarray(is_known, dtype=bool)

    if scores.shape != (len(probe_ids), len(gallery_ids)):
        raise ValueError("score_matrix shape does not match probe/gallery IDs")

    gallery_index = {s: i for i, s in enumerate(gallery_ids.tolist())}
    known_indices = np.where(known)[0]
    unknown_indices = np.where(~known)[0]

    genuine = []
    impostor = []
    correct = []
    top1_margin = []
    true_margin = []
    ranks = []

    for p in range(len(probe_ids)):
        row = scores[p]
        order = np.argsort(row)[::-1]
        if len(row) >= 2:
            top1_margin.append(float(row[order[0]] - row[order[1]]))

        if known[p]:
            if probe_ids[p] not in gallery_index:
                raise ValueError(
                    f"Known probe speaker {probe_ids[p]!r} not found in gallery."
                )
            gi = gallery_index[probe_ids[p]]
            genuine.append(float(row[gi]))

            imp_mask = np.ones(len(gallery_ids), dtype=bool)
            imp_mask[gi] = False
            imp = row[imp_mask]
            impostor.extend(imp.tolist())

            best_imp = float(np.max(imp)) if len(imp) else float("-inf")
            true_margin.append(float(row[gi] - best_imp))
            pred = int(order[0])
            correct.append(pred == gi)
            rank = int(np.where(order == gi)[0][0]) + 1
            ranks.append(rank)
        else:
            impostor.extend(row.tolist())

    # Cap pair-level impostors for memory/statistics if gallery is very large.
    impostor = np.asarray(impostor, dtype=float)
    if len(impostor) > max_impostor_scores:
        rng = np.random.default_rng(seed)
        impostor = impostor[
            rng.choice(len(impostor), size=max_impostor_scores, replace=False)
        ]
    genuine = np.asarray(genuine, dtype=float)

    result = {
        "known_probe_count": int(len(known_indices)),
        "unknown_probe_count": int(len(unknown_indices)),
        "gallery_speaker_count": int(len(gallery_ids)),
        "genuine_score_mean": _safe_mean(genuine),
        "genuine_score_std": _safe_std(genuine),
        "impostor_score_mean": _safe_mean(impostor),
        "impostor_score_std": _safe_std(impostor),
        "d_prime": dprime(genuine, impostor),
        "top1_accuracy_known": _safe_mean(correct),
        "top1_top2_margin_mean_all_probes": _safe_mean(top1_margin),
        "true_vs_best_impostor_margin_mean_known": _safe_mean(true_margin),
        "true_vs_best_impostor_margin_median_known": (
            float(np.median(true_margin)) if true_margin else float("nan")
        ),
    }

    for k in (1, 5, 10):
        result[f"top{k}_accuracy_known"] = (
            float(np.mean(np.asarray(ranks) <= k)) if ranks else float("nan")
        )

    eer, eer_threshold = eer_from_scores(genuine, impostor)
    result["pair_score_eer"] = eer
    result["pair_score_eer_threshold"] = eer_threshold

    if len(genuine) and len(impostor):
        y = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
        s = np.concatenate([genuine, impostor])
        result["pair_score_roc_auc"] = float(roc_auc_score(y, s))
    else:
        result["pair_score_roc_auc"] = float("nan")

    if len(unknown_indices):
        unknown_max = scores[unknown_indices].max(axis=1)
        known_max = scores[known_indices].max(axis=1) if len(known_indices) else np.array([])
        known_pred = scores[known_indices].argmax(axis=1) if len(known_indices) else np.array([])
        known_true = np.array(
            [gallery_index[probe_ids[p]] for p in known_indices], dtype=int
        ) if len(known_indices) else np.array([])

        dir_far = {}
        for far in far_targets:
            threshold = _threshold_for_target_far(unknown_max, far)
            empirical_far = float(np.mean(unknown_max >= threshold))
            if len(known_indices):
                dir_value = float(
                    np.mean((known_pred == known_true) & (known_max >= threshold))
                )
            else:
                dir_value = float("nan")
            dir_far[str(far)] = {
                "threshold": threshold,
                "empirical_far": empirical_far,
                "dir": dir_value,
            }
        result["dir_at_far"] = dir_far
    else:
        result["dir_at_far"] = {}

    return result
