import numpy as np
import mne


def bandpass_filter(epochs, l_freq=8, h_freq=30):
    return epochs.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=False)


def standardize_epochs(X):
    """
    Z-score normalization per channel per epoch.
    X: (N, C, T)
    """
    X_std = np.zeros_like(X)
    for i in range(X.shape[0]):
        mean = X[i].mean(axis=1, keepdims=True)
        std = X[i].std(axis=1, keepdims=True) + 1e-8
        X_std[i] = (X[i] - mean) / std
    return X_std