import numpy as np
from scipy.stats import ttest_rel, t


def paired_ttest(x, y):
    return ttest_rel(x, y)


def cohens_d(x, y):
    diff = np.array(x) - np.array(y)
    return diff.mean() / diff.std(ddof=1)


def confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))
    margin = t.ppf((1 + confidence) / 2., len(data) - 1) * sem
    return mean - margin, mean + margin