from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax.

    embeddings are expected to be L2 normalized.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        cosine = F.linear(
            F.normalize(embeddings, dim=1),
            F.normalize(self.weight, dim=1),
        )
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target_cosine = torch.cos(theta + self.margin)

        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(cosine.dtype)
        logits = cosine * (1.0 - one_hot) + target_cosine * one_hot
        return logits * self.scale
