from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.net(x)


class Res2Conv1d(nn.Module):
    """Multi-scale temporal processing inspired by Res2Net/ECAPA."""

    def __init__(self, channels: int, scale: int = 8, kernel_size: int = 3, dilation: int = 2):
        super().__init__()
        if channels % scale != 0:
            raise ValueError("channels must be divisible by scale")
        self.scale = scale
        width = channels // scale
        self.blocks = nn.ModuleList(
            [TDNNBlock(width, width, kernel_size, dilation) for _ in range(scale - 1)]
        )

    def forward(self, x):
        chunks = torch.chunk(x, self.scale, dim=1)
        outputs = [chunks[0]]
        running = chunks[1]
        outputs.append(self.blocks[0](running))
        for i in range(2, self.scale):
            running = chunks[i] + outputs[-1]
            outputs.append(self.blocks[i - 1](running))
        return torch.cat(outputs, dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.fc2 = nn.Conv1d(bottleneck, channels, kernel_size=1)

    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class SERes2Block(nn.Module):
    def __init__(
        self,
        channels: int = 512,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 2,
        se_bottleneck: int = 128,
    ):
        super().__init__()
        self.pre = TDNNBlock(channels, channels, kernel_size=1)
        self.res2 = Res2Conv1d(channels, scale, kernel_size, dilation)
        self.post = TDNNBlock(channels, channels, kernel_size=1)
        self.se = SEBlock(channels, se_bottleneck)

    def forward(self, x):
        y = self.pre(x)
        y = self.res2(y)
        y = self.post(y)
        y = self.se(y)
        return x + y


class AttentiveStatsPool(nn.Module):
    """
    Channel-dependent attentive statistics pooling.
    Input [B, C, T] -> output [B, 2C].
    """

    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, attention_channels, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, channels, 1),
        )

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(
            (x.pow(2).mean(dim=-1, keepdim=True) - mean.pow(2)).clamp_min(1e-5)
        )
        global_context = torch.cat(
            [x, mean.expand_as(x), std.expand_as(x)], dim=1
        )
        alpha = torch.softmax(self.attention(global_context), dim=-1)

        mu = torch.sum(alpha * x, dim=-1)
        second = torch.sum(alpha * x.pow(2), dim=-1)
        sigma = torch.sqrt((second - mu.pow(2)).clamp_min(1e-5))
        return torch.cat([mu, sigma], dim=1)


class SpeakerEncoder(nn.Module):
    """
    ECAPA-style speaker encoder.

    Expected input: [B, n_mels, T]
    Output: L2-normalized speaker embedding [B, embedding_dim]
    """

    def __init__(
        self,
        n_mels: int = 80,
        channels: int = 512,
        scale: int = 8,
        se_bottleneck: int = 128,
        attention_channels: int = 128,
        embedding_dim: int = 192,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.channels = channels
        self.embedding_dim = embedding_dim

        self.input_tdnn = TDNNBlock(n_mels, channels, kernel_size=5, dilation=1)
        self.block1 = SERes2Block(channels, scale, 3, 2, se_bottleneck)
        self.block2 = SERes2Block(channels, scale, 3, 3, se_bottleneck)
        self.block3 = SERes2Block(channels, scale, 3, 4, se_bottleneck)

        self.mfa = TDNNBlock(channels * 3, channels * 3, kernel_size=1)
        self.pool = AttentiveStatsPool(channels * 3, attention_channels)
        self.pool_bn = nn.BatchNorm1d(channels * 6)
        self.embedding = nn.Linear(channels * 6, embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected [B,F,T], got {tuple(x.shape)}")
        if x.shape[1] != self.n_mels:
            raise ValueError(
                f"Expected {self.n_mels} feature bins, got {x.shape[1]}"
            )

        x = self.input_tdnn(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        x = self.mfa(torch.cat([x1, x2, x3], dim=1))
        x = self.pool(x)
        x = self.pool_bn(x)
        x = self.embedding(x)
        x = self.embedding_bn(x)
        return F.normalize(x, p=2, dim=1)
