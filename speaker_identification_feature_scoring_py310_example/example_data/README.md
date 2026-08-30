# Bundled Example Training Data

This folder contains actual 16-kHz mono WAV files so the project can be run
immediately after dependency installation.

## Important

The WAV files are **synthetic speech-like signals**, not recordings of real
people. They are included only to demonstrate:

- dataset loading,
- deep speaker-embedding training,
- enrollment,
- cosine feature scoring,
- genuine/impostor score analysis,
- open-set unknown-speaker evaluation,
- threshold calibration.

**Do not use the example accuracy as evidence of real-world speaker-identification
performance.**

## Speaker layout

Known speakers:

```text
spk001
spk002
spk003
spk004
```

Each known speaker has:

```text
6 training WAVs
2 development WAVs
2 enrollment WAVs
2 probe WAVs
```

Unknown probe-only identities:

```text
unknown_a
unknown_b
```

Each unknown identity has 2 probe WAVs.

Total:

```text
24 training WAVs
 8 development WAVs
 8 enrollment WAVs
12 probe WAVs
----------------
52 WAV files
```

All WAV files are:

```text
sample rate: 16000 Hz
channels:    1
PCM:         signed 16-bit
duration:    approximately 2.4 s
```

## Manifests

`train.csv`

```csv
path,speaker_id
audio/train/spk001_train_01.wav,spk001
...
```

`dev.csv`

```csv
path,speaker_id
audio/dev/spk001_dev_01.wav,spk001
...
```

`enroll.csv`

```csv
path,speaker_id
audio/enroll/spk001_enroll_01.wav,spk001
...
```

`probes.csv`

```csv
path,speaker_id,is_known
audio/probe/spk001_probe_01.wav,spk001,1
audio/probe/unknown_a_probe_01.wav,unknown_a,0
...
```

## Quick run

From the project root:

```powershell
.\run_example.ps1
```

The lightweight `config_example.json` uses a smaller encoder than the main
`config.json`, so it is suitable for demonstrating the pipeline on CPU.

The bundled WAVs can be regenerated deterministically with:

```powershell
python scripts\generate_example_data.py
```
