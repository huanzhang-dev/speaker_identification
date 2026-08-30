from __future__ import annotations

from pathlib import Path
import random

import torch
import torch.nn as nn
import torchaudio


def load_audio(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio as mono float tensor [1, N] and resample if required."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.ndim != 2:
        raise RuntimeError(f"Unexpected waveform shape for {path}: {tuple(waveform.shape)}")
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform


def fix_waveform_length(
    waveform: torch.Tensor,
    target_samples: int,
    training: bool,
) -> torch.Tensor:
    """Random/center crop or zero-pad to a fixed number of samples."""
    n = waveform.shape[-1]
    if n == target_samples:
        return waveform
    if n > target_samples:
        max_start = n - target_samples
        start = random.randint(0, max_start) if training else max_start // 2
        return waveform[..., start:start + target_samples]
    return torch.nn.functional.pad(waveform, (0, target_samples - n))


def random_gain(waveform: torch.Tensor, min_db: float = -6.0, max_db: float = 6.0):
    gain_db = random.uniform(min_db, max_db)
    return waveform * (10.0 ** (gain_db / 20.0))


def add_gaussian_noise(
    waveform: torch.Tensor,
    min_snr_db: float = 10.0,
    max_snr_db: float = 30.0,
):
    """Simple self-contained noise augmentation. Real noise/RIR is better when available."""
    snr_db = random.uniform(min_snr_db, max_snr_db)
    signal_power = waveform.pow(2).mean().clamp_min(1e-10)
    noise = torch.randn_like(waveform)
    noise_power = noise.pow(2).mean().clamp_min(1e-10)
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = noise * torch.sqrt(target_noise_power / noise_power)
    return waveform + noise


class LogMelFrontend(nn.Module):
    """
    16-kHz log-Mel frontend.
    Output shape: [n_mels, T].
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 20.0,
        f_max: float = 7600.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            center=True,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [1, N] or [B, N]
        mel = self.mel(waveform)
        logmel = torch.log(mel.clamp_min(1e-6))
        # Per-utterance cepstral mean normalization across time.
        logmel = logmel - logmel.mean(dim=-1, keepdim=True)
        return logmel
