from __future__ import annotations

"""
Regenerate the bundled synthetic example WAV dataset.

The included WAV files are synthetic, speech-like signals created only for
pipeline demonstration. They are NOT suitable for benchmarking a real
speaker-identification model.
"""

from pathlib import Path
import math
import wave

import numpy as np

FS = 16000
DURATION = 2.4

SPEAKERS = {
    "spk001": {"f0": 112.0, "formant_scale": 0.92, "tilt": 1.10, "seed": 101},
    "spk002": {"f0": 142.0, "formant_scale": 1.00, "tilt": 1.25, "seed": 202},
    "spk003": {"f0": 178.0, "formant_scale": 1.08, "tilt": 1.35, "seed": 303},
    "spk004": {"f0": 218.0, "formant_scale": 1.16, "tilt": 1.50, "seed": 404},
}
UNKNOWN_SPEAKERS = {
    "unknown_a": {"f0": 128.0, "formant_scale": 1.13, "tilt": 1.18, "seed": 505},
    "unknown_b": {"f0": 198.0, "formant_scale": 0.95, "tilt": 1.42, "seed": 606},
}
VOWELS = {
    "a": (730, 1090, 2440),
    "e": (530, 1840, 2480),
    "i": (270, 2290, 3010),
    "o": (570, 840, 2410),
    "u": (300, 870, 2240),
}

def synth_segment(rng, f0, formants, duration, tilt, formant_scale):
    n = max(1, int(round(duration * FS)))
    t = np.arange(n, dtype=np.float64) / FS
    vibrato_rate = rng.uniform(3.2, 5.5)
    vibrato_depth = rng.uniform(0.006, 0.018)
    f0_inst = f0 * (1.0 + vibrato_depth * np.sin(2*np.pi*vibrato_rate*t))
    phase = 2*np.pi*np.cumsum(f0_inst) / FS
    y = np.zeros(n, dtype=np.float64)
    formants = np.asarray(formants, dtype=np.float64) * formant_scale
    bandwidths = np.array([90.0, 130.0, 180.0], dtype=np.float64)
    max_h = int((FS * 0.46) // f0)
    for h in range(1, max_h + 1):
        freq = h * f0
        harmonic_amp = 1.0 / (h ** tilt)
        formant_weight = 0.12
        for fk, bw in zip(formants, bandwidths):
            formant_weight += math.exp(-0.5 * ((freq - fk) / bw) ** 2)
        y += harmonic_amp * formant_weight * np.sin(
            h * phase + rng.uniform(0, 2*np.pi)
        )
    breath = rng.normal(0.0, 1.0, n)
    breath = np.convolve(breath, np.ones(5)/5.0, mode="same")
    y += 0.018 * breath
    env = np.sin(np.pi * np.linspace(0.0, 1.0, n)) ** 0.7
    return y * env

def synth_utterance(profile, utterance_index, duration=DURATION):
    rng = np.random.default_rng(profile["seed"] + utterance_index * 7919)
    keys = list(VOWELS.keys())
    rng.shuffle(keys)
    segments = []
    remaining = duration
    idx = 0
    while remaining > 0.08:
        vowel = keys[idx % len(keys)]
        seg_dur = min(rng.uniform(0.22, 0.46), remaining)
        segments.append(synth_segment(
            rng,
            profile["f0"] * rng.uniform(0.96, 1.04),
            VOWELS[vowel],
            seg_dur,
            profile["tilt"],
            profile["formant_scale"],
        ))
        remaining -= seg_dur
        idx += 1
    y = np.concatenate(segments)
    target_n = int(duration * FS)
    y = np.pad(y, (0, max(0, target_n-len(y))))[:target_n]
    a = rng.uniform(0.15, 0.45)
    colored = np.empty_like(y)
    prev = 0.0
    for i, v in enumerate(y):
        prev = (1-a)*v + a*prev
        colored[i] = prev
    y = colored
    signal_rms = np.sqrt(np.mean(y*y) + 1e-12)
    noise = rng.normal(0.0, 1.0, target_n)
    snr_db = rng.uniform(24.0, 36.0)
    y += noise * signal_rms / (10**(snr_db/20.0))
    y = 0.82 * y / (np.max(np.abs(y)) + 1e-12)
    return y.astype(np.float32)

def write_wav(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(pcm.tobytes())

def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")

def main():
    root = Path(__file__).resolve().parents[1] / "example_data"
    train_rows, dev_rows, enroll_rows, probe_rows = [], [], [], []

    for sid, profile in SPEAKERS.items():
        for i in range(6):
            rel = Path("audio")/"train"/f"{sid}_train_{i+1:02d}.wav"
            write_wav(root/rel, synth_utterance(profile, 10+i))
            train_rows.append((rel.as_posix(), sid))
        for i in range(2):
            rel = Path("audio")/"dev"/f"{sid}_dev_{i+1:02d}.wav"
            write_wav(root/rel, synth_utterance(profile, 30+i))
            dev_rows.append((rel.as_posix(), sid))
        for i in range(2):
            rel = Path("audio")/"enroll"/f"{sid}_enroll_{i+1:02d}.wav"
            write_wav(root/rel, synth_utterance(profile, 50+i))
            enroll_rows.append((rel.as_posix(), sid))
        for i in range(2):
            rel = Path("audio")/"probe"/f"{sid}_probe_{i+1:02d}.wav"
            write_wav(root/rel, synth_utterance(profile, 70+i))
            probe_rows.append((rel.as_posix(), sid, 1))

    for sid, profile in UNKNOWN_SPEAKERS.items():
        for i in range(2):
            rel = Path("audio")/"probe"/f"{sid}_probe_{i+1:02d}.wav"
            write_wav(root/rel, synth_utterance(profile, 90+i))
            probe_rows.append((rel.as_posix(), sid, 0))

    write_csv(root/"train.csv", ["path","speaker_id"], train_rows)
    write_csv(root/"dev.csv", ["path","speaker_id"], dev_rows)
    write_csv(root/"enroll.csv", ["path","speaker_id"], enroll_rows)
    write_csv(root/"probes.csv", ["path","speaker_id","is_known"], probe_rows)
    print("Regenerated example data at", root)

if __name__ == "__main__":
    main()
