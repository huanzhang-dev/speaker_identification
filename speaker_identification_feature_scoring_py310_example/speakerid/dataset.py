from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import random

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .audio import (
    LogMelFrontend,
    add_gaussian_noise,
    fix_waveform_length,
    load_audio,
    random_gain,
)


class SpeakerDataset(Dataset):
    """
    Manifest columns:
        path,speaker_id

    Paths may be absolute or relative to data_root.
    """

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        speaker_to_index: dict[str, int],
        feature_config: dict,
        segment_seconds: float = 3.0,
        training: bool = True,
        noise_probability: float = 0.35,
        gain_probability: float = 0.35,
        freq_mask: int = 8,
        time_mask: int = 20,
    ):
        self.df = pd.read_csv(manifest)
        required = {"path", "speaker_id"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"Manifest must contain columns {sorted(required)}")

        self.data_root = Path(data_root)
        self.speaker_to_index = speaker_to_index
        self.feature_config = feature_config
        self.frontend = LogMelFrontend(**feature_config)
        self.sample_rate = feature_config["sample_rate"]
        self.target_samples = int(segment_seconds * self.sample_rate)
        self.training = training
        self.noise_probability = noise_probability
        self.gain_probability = gain_probability

        import torchaudio
        self.freq_masker = (
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask)
            if training and freq_mask > 0 else None
        )
        self.time_masker = (
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
            if training and time_mask > 0 else None
        )

        unknown = set(self.df["speaker_id"].astype(str)) - set(speaker_to_index)
        if unknown:
            raise ValueError(
                f"Manifest contains speaker IDs missing from mapping: {sorted(unknown)[:5]}"
            )

        self.labels = [
            self.speaker_to_index[str(v)] for v in self.df["speaker_id"].astype(str)
        ]

    def __len__(self):
        return len(self.df)

    def _resolve(self, value):
        p = Path(str(value))
        return p if p.is_absolute() else self.data_root / p

    def __getitem__(self, index):
        row = self.df.iloc[index]
        waveform = load_audio(self._resolve(row["path"]), self.sample_rate)
        waveform = fix_waveform_length(
            waveform, self.target_samples, training=self.training
        )

        if self.training:
            if random.random() < self.gain_probability:
                waveform = random_gain(waveform)
            if random.random() < self.noise_probability:
                waveform = add_gaussian_noise(waveform)

        with torch.no_grad():
            feat = self.frontend(waveform).squeeze(0)  # [F,T]

        if self.freq_masker is not None:
            feat = self.freq_masker(feat)
        if self.time_masker is not None:
            feat = self.time_masker(feat)

        label = self.speaker_to_index[str(row["speaker_id"])]
        return feat, torch.tensor(label, dtype=torch.long)


class SpeakerBalancedBatchSampler(Sampler[list[int]]):
    """
    Each batch contains S speakers x U utterances per speaker.

    This is preferable to pure random utterance sampling for speaker losses.
    """

    def __init__(
        self,
        labels: list[int],
        speakers_per_batch: int = 16,
        utterances_per_speaker: int = 2,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ):
        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker
        self.seed = seed
        self.epoch = 0

        by_speaker = defaultdict(list)
        for idx, label in enumerate(labels):
            by_speaker[int(label)].append(idx)
        self.by_speaker = dict(by_speaker)
        self.speakers = sorted(self.by_speaker)

        if len(self.speakers) < 2:
            raise ValueError("Speaker embedding training requires at least 2 speakers.")
        # Adapt to small development projects instead of failing because the
        # configured speaker count is larger than the available roster.
        self.speakers_per_batch = min(speakers_per_batch, len(self.speakers))

        batch_size = self.speakers_per_batch * utterances_per_speaker
        self.batches_per_epoch = (
            batches_per_epoch
            if batches_per_epoch is not None
            else max(1, math.ceil(len(labels) / batch_size))
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            chosen_speakers = rng.sample(self.speakers, self.speakers_per_batch)
            batch = []
            for spk in chosen_speakers:
                candidates = self.by_speaker[spk]
                if len(candidates) >= self.utterances_per_speaker:
                    selected = rng.sample(candidates, self.utterances_per_speaker)
                else:
                    selected = [
                        rng.choice(candidates)
                        for _ in range(self.utterances_per_speaker)
                    ]
                batch.extend(selected)
            rng.shuffle(batch)
            yield batch
