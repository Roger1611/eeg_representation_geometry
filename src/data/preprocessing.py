import mne


def bandpass_filter(epochs, l_freq=8.0, h_freq=30.0):
    return epochs.copy().filter(l_freq=l_freq, h_freq=h_freq)


def standardize_epochs(X):
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X - mean) / std