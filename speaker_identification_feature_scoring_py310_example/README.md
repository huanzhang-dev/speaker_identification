# Deep Speaker Identification with Feature-Based Scoring — Python 3.10.2

This project is a **Python 3.10.2-compatible** implementation of an
embedding-based deep speaker-identification system.

The architecture is intentionally based on feature scoring rather than only
on an N-class classifier:

```text
16-kHz speech
    -> 80-bin log-Mel features
    -> ECAPA-style speaker encoder
    -> L2-normalized speaker embedding
    -> feature-based gallery scoring
    -> Speaker ID / Unknown
```

Training uses **AAM-Softmax** to shape a discriminative embedding space.
Deployment uses speaker enrollment prototypes and similarity scoring.

## Python compatibility

Target interpreter:

```text
Python 3.10.2
```

Pinned core packages:

| Package | Version |
|---|---:|
| torch | 2.2.2 |
| torchaudio | 2.2.2 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| scikit-learn | 1.5.1 |
| matplotlib | 3.8.4 |
| pytest | 8.2.2 |

The code avoids Python 3.11/3.12-only language features and avoids newer
PyTorch-only checkpoint arguments.

## Windows installation

### CPU

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_windows_cpu.ps1
```

### NVIDIA CUDA 12.1 wheel set

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_windows_cuda121.ps1
```

If your NVIDIA driver/CUDA environment requires another PyTorch wheel set,
install the corresponding PyTorch/torchaudio pair first, then install the
remaining pinned packages.

Verify the environment:

```powershell
python scripts\check_environment.py
```

### Project-local CPU environment

Create the `venv` environment with the CPU-only PyTorch build:

```powershell
.\setup_venv.cmd
```

Run training on the CPU, passing the normal training arguments after the
launcher name:

```powershell
.\train_cpu.cmd --train-manifest example_data\train.csv --dev-manifest example_data\dev.csv --data-root example_data --config example_data\config_example.json --output-dir runs\cpu
```

This setup intentionally installs no CUDA runtime, and `train.py` explicitly
uses the CPU even if the host computer has a GPU.

Run the project smoke test:

```powershell
python scripts\smoke_test.py
```



## Bundled runnable example data

This edition includes **52 actual WAV files** under `example_data/`:

```text
example_data/
├── config_example.json
├── train.csv
├── dev.csv
├── enroll.csv
├── probes.csv
└── audio/
    ├── train/
    ├── dev/
    ├── enroll/
    └── probe/
```

There are four known synthetic demo speakers and two unknown probe-only
speakers. The data is 16-kHz mono PCM and is intended only to verify the
software pipeline.

After installing dependencies, run:

```powershell
.\run_example.ps1
```

This trains a small example encoder, builds the gallery, evaluates
feature-based scores, and calibrates an example unknown-speaker threshold.

## Architecture

```text
16-kHz waveform
      |
      v
80-bin log-Mel
      |
      v
TDNN (512 channels)
      |
      +--> SE-Res2 block, dilation 2
      +--> SE-Res2 block, dilation 3
      +--> SE-Res2 block, dilation 4
      |
      v
Multi-layer feature aggregation
      |
      v
Attentive statistics pooling
      |
      v
Linear -> 192-D speaker embedding
      |
      v
L2 normalization
      |
      +-----------------------+
      |                       |
training                  deployment
      |                       |
AAM-Softmax             cosine scoring
speaker labels          gallery prototypes
                              |
                              v
                       Speaker ID / Unknown
```

## Feature-based scoring implemented

The evaluator calculates:

- mean genuine cosine score
- genuine score standard deviation
- mean impostor cosine score
- impostor score standard deviation
- **d-prime (`d'`)**
- pair-score EER
- pair-score ROC-AUC
- mean Top-1/Top-2 decision margin
- mean true-speaker vs best-impostor margin
- within-speaker cosine distance
- between-speaker centroid cosine distance
- between/within separation ratio
- Fisher trace ratio
- cosine silhouette score
- Top-1 / Top-5 / Top-10 known-speaker accuracy
- **DIR@FAR** for open-set identification

### d-prime

\[
d'=
\frac{|\mu_G-\mu_I|}
{\sqrt{\frac{1}{2}(\sigma_G^2+\sigma_I^2)}}
\]

where `G` is the genuine-score distribution and `I` is the impostor-score distribution.

Higher is better.

### Within-speaker distance

For speaker centroid \(c_i\):

\[
D_W =
\frac{1}{N}
\sum_i\sum_j
\left(1-\cos(z_{ij},c_i)\right)
\]

Lower is better.

### Between-speaker centroid distance

\[
D_B =
\frac{2}{K(K-1)}
\sum_{i<j}
\left(1-\cos(c_i,c_j)\right)
\]

Higher is better.

### Separation ratio

\[
R_{\mathrm{sep}}=\frac{D_B}{D_W+\epsilon}
\]

Higher is better.

### Identification margin

For the highest and second-highest gallery scores:

\[
M=S_{(1)}-S_{(2)}
\]

The evaluator also computes a stronger diagnostic for known probes:

\[
M_{\mathrm{true}}
=
S_{\mathrm{true}}
-
\max_{j\ne \mathrm{true}}S_j
\]

Positive and large is desirable.

## Project structure

```text
speaker_identification_feature_scoring/
├── config.json
├── requirements.txt
├── README.md
├── data/
│   ├── train_manifest_example.csv
│   ├── enroll_manifest_example.csv
│   └── probe_manifest_example.csv
├── speakerid/
│   ├── __init__.py
│   ├── audio.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── scoring.py
├── scripts/
│   ├── train.py
│   ├── enroll.py
│   ├── identify.py
│   └── evaluate.py
└── tests/
    └── test_shapes.py
```

# 1. Install

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

# 2. Prepare training data

Create separate train and development manifests.

Example:

```csv
path,speaker_id
audio/spk001_001.wav,spk001
audio/spk001_002.wav,spk001
audio/spk002_001.wav,spk002
audio/spk002_002.wav,spk002
```

The development manifest must contain speaker IDs present in training, because the training loss is supervised speaker classification.

**Important:** split by recording/session, not by random fragments from the same recording.

# 3. Train

```powershell
python scripts/train.py `
    --config config.json `
    --train-manifest dataset/train.csv `
    --dev-manifest dataset/dev.csv `
    --data-root dataset `
    --output-dir runs/exp01
```

Output:

```text
runs/exp01/
├── best_encoder.pt
├── last.pt
├── history.json
└── speaker_mapping.json
```

The deployment model is `best_encoder.pt`.

## Transfer learning

If you already have a compatible pretrained encoder checkpoint:

```powershell
python scripts/train.py `
    --train-manifest dataset/train.csv `
    --dev-manifest dataset/dev.csv `
    --data-root dataset `
    --output-dir runs/exp02 `
    --pretrained-encoder pretrained_encoder.pt
```

Only matching model keys are loaded. The new AAM speaker head is created for the training speakers in the new manifest.

# 4. Enroll speakers

Enrollment manifest:

```csv
path,speaker_id
audio/enroll_spk001_a.wav,spk001
audio/enroll_spk001_b.wav,spk001
audio/enroll_spk002_a.wav,spk002
audio/enroll_spk002_b.wav,spk002
```

Build the gallery:

```powershell
python scripts/enroll.py `
    --checkpoint runs/exp01/best_encoder.pt `
    --manifest dataset/enroll.csv `
    --data-root dataset `
    --output runs/exp01/gallery.npz
```

For each speaker, normalized utterance embeddings are averaged and normalized again to create a prototype.

Use multiple enrollment utterances whenever possible.

# 5. Identify one speaker

```powershell
python scripts/identify.py `
    --checkpoint runs/exp01/best_encoder.pt `
    --gallery runs/exp01/gallery.npz `
    --audio dataset/audio/test.wav `
    --threshold 0.55 `
    --topk 5
```

Example output:

```text
Top matches:
 1. spk001  cosine=0.812341
 2. spk014  cosine=0.561202
 3. spk021  cosine=0.493820

Top1-Top2 margin: 0.251139
Decision: spk001
```

If the highest score is below the threshold:

```text
Decision: UNKNOWN
```

`0.55` is only an example threshold. It must be calibrated with an open-set development set.

# 6. Feature-score evaluation

Probe manifest:

```csv
path,speaker_id,is_known
audio/probe_spk001.wav,spk001,1
audio/probe_spk002.wav,spk002,1
audio/probe_unknown_001.wav,unknown_a,0
audio/probe_unknown_002.wav,unknown_b,0
```

Evaluate:

```powershell
python scripts/evaluate.py `
    --checkpoint runs/exp01/best_encoder.pt `
    --enroll-manifest dataset/enroll.csv `
    --probe-manifest dataset/probes.csv `
    --data-root dataset `
    --output-dir runs/exp01/evaluation
```

Outputs:

```text
evaluation/
├── metrics.json
├── probe_scores.csv
└── genuine_impostor_scores.png
```

`metrics.json` contains feature geometry, score separation, Top-k identification, EER, and DIR@FAR.

# 7. Interpreting the feature scores

A desirable result looks qualitatively like:

```text
genuine mean                    high
impostor mean                   low
d-prime                         high
within-speaker distance         low
between-speaker distance        high
separation ratio                high
Fisher ratio                    high
silhouette                      high
Top1-Top2 margin                high
true-vs-impostor margin         positive and high
EER                             low
Top-1 accuracy                  high
DIR@target FAR                  high
```

Do not define universal "good" cosine values such as 0.7 or 0.8. The score scale depends on:

- model
- training corpus
- augmentation
- duration
- microphones
- language
- enrollment count
- gallery size

Thresholds must be calibrated on a development set representative of deployment.

# 8. Recommended experiment sequence

Keep the same train/dev/evaluation split.

```text
E01  Default encoder + AAM-Softmax
E02  embedding_dim = 256
E03  channels = 768
E04  margin = 0.3
E05  segment_seconds = 2
E06  segment_seconds = 4
E07  disable simple Gaussian augmentation
E08  replace simple noise with real MUSAN/RIR-style augmentation
E09  1 enrollment utterance
E10  3 enrollment utterances
E11  5 enrollment utterances
E12  threshold calibration at FAR = 1%, 0.1%
```

For every experiment compare not only Top-1 but:

```text
d'
within distance
between distance
separation ratio
true-speaker margin
EER
DIR@FAR
```

# 9. Recommended production improvements

The project is intentionally self-contained. For a production system, add:

- real noise and room-impulse-response augmentation
- robust VAD
- audio-quality scoring
- separate anti-spoof / deepfake detector
- score calibration
- approximate-nearest-neighbor search for very large galleries
- model/template versioning
- confidence intervals and subgroup testing
- enrollment identity verification
- secure biometric-template storage

# 10. Why embedding-based identification?

A direct N-class classifier ties the model to the training roster.

This project instead learns:

\[
x \rightarrow z_{\mathrm{speaker}}
\]

and identification is:

\[
\hat{s}
=
\arg\max_i
\cos(z_{\mathrm{probe}},g_i)
\]

with open-set rejection:

\[
\hat{s}
=
\begin{cases}
\arg\max_i S_i,&\max_iS_i\ge\tau\\
\mathrm{UNKNOWN},&\max_iS_i<\tau
\end{cases}
\]

That lets you add or remove enrolled speakers without retraining the encoder.


## 11. Comparing feature-scoring functions

Cosine is the recommended default because the encoder produces L2-normalized
embeddings and AAM-Softmax trains angular separation.

You can compare Euclidean scoring without retraining:

```powershell
python scripts/evaluate.py `
    --checkpoint runs/exp01/best_encoder.pt `
    --enroll-manifest dataset/enroll.csv `
    --probe-manifest dataset/probes.csv `
    --data-root dataset `
    --score-method euclidean `
    --output-dir runs/exp01/evaluation_euclidean
```

The library also implements **Mahalanobis scoring**:

```python
from speakerid.scoring import (
    fit_mahalanobis_precision,
    mahalanobis_score_matrix,
)

precision = fit_mahalanobis_precision(background_embeddings)
scores = mahalanobis_score_matrix(
    probe_embeddings,
    gallery_embeddings,
    precision,
)
```

The covariance/precision matrix should be estimated from an independent
training/background cohort, not from the test probes.

All score matrices in this project follow one convention:

```text
higher score = more likely to be the same speaker
```

Therefore Euclidean and Mahalanobis distances are returned as negative
distances.

PLDA is deliberately not hard-coded into the minimal project because a proper
PLDA backend requires a separately trained within-/between-speaker generative
model and careful domain adaptation. It can be added as an experimental backend
after the cosine baseline is stable.


## Open-set threshold calibration

Do not use an arbitrary fixed cosine threshold in deployment. First run
`scripts/evaluate.py` with known and unknown development probes. Then calibrate
the threshold for a required false-alarm rate:

```powershell
python scripts\calibrate_threshold.py `
    --probe-scores runs/exp01/evaluation/probe_scores.csv `
    --target-far 0.01 `
    --output runs/exp01/threshold_far_1pct.json
```

This produces a threshold selected from **unknown development speakers**.
Use an evaluation/test set only once after model and threshold choices are
finished.
