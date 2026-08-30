from __future__ import annotations

import platform
import sys


MIN_VERSION = (3, 10, 2)
MAX_VERSION = (3, 11, 0)


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:   ", platform.python_version())
    print("Platform:         ", platform.platform())

    current = sys.version_info[:3]
    if current < MIN_VERSION:
        raise SystemExit(
            "ERROR: Python 3.10.2 or newer in the 3.10 series is required."
        )
    if current >= MAX_VERSION:
        print(
            "WARNING: This project is pinned for Python 3.10.x. "
            "Your interpreter is newer; it may still work, but the pinned "
            "compatibility target is Python 3.10.2."
        )

    import torch
    import torchaudio
    import numpy
    import pandas
    import sklearn
    import matplotlib

    print("torch:            ", torch.__version__)
    print("torchaudio:       ", torchaudio.__version__)
    print("numpy:            ", numpy.__version__)
    print("pandas:           ", pandas.__version__)
    print("scikit-learn:     ", sklearn.__version__)
    print("matplotlib:       ", matplotlib.__version__)
    print("CUDA available:   ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA runtime:     ", torch.version.cuda)
        print("GPU:              ", torch.cuda.get_device_name(0))

    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
