from __future__ import annotations

import numpy as np
import torch

from speakerid.losses import AAMSoftmax
from speakerid.metrics import feature_geometry_metrics, evaluate_gallery_scores
from speakerid.model import SpeakerEncoder
from speakerid.scoring import build_gallery, cosine_score_matrix


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    model = SpeakerEncoder(
        n_mels=80,
        channels=128,
        scale=8,
        se_bottleneck=32,
        attention_channels=32,
        embedding_dim=64,
    )
    x = torch.randn(6, 80, 300)
    z = model(x)

    assert z.shape == (6, 64)
    assert torch.allclose(z.norm(dim=1), torch.ones(6), atol=1e-4)

    head = AAMSoftmax(64, 3, margin=0.2, scale=30.0)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    logits = head(z, labels)
    assert logits.shape == (6, 3)

    z_np = z.detach().numpy()
    speaker_ids = ["A", "A", "B", "B", "C", "C"]
    gallery_ids, gallery = build_gallery(speaker_ids, z_np)

    scores = cosine_score_matrix(z_np, gallery)
    metrics = evaluate_gallery_scores(
        scores,
        speaker_ids,
        gallery_ids.tolist(),
        [True] * len(speaker_ids),
    )
    geometry = feature_geometry_metrics(z_np, speaker_ids)

    print("Encoder output:", z.shape)
    print("AAM logits:    ", logits.shape)
    print("Gallery shape: ", gallery.shape)
    print("Top-1 accuracy:", metrics["top1_accuracy_known"])
    print("d-prime:       ", metrics["d_prime"])
    print("Fisher ratio:  ", geometry["fisher_trace_ratio"])
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
