import numpy as np
from pathlib import Path


def load_physionet_processed(path: Path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(int)
    return X, y