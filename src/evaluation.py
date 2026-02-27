import numpy as np
from sklearn.metrics import accuracy_score


def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def compute_class_centroids(embeddings, labels):
    classes = np.unique(labels)
    centroids = {}
    for c in classes:
        centroids[c] = embeddings[labels == c].mean(axis=0)
    return centroids


def compute_ess(embeddings, labels):
    """
    Embedding Stability Score (mean intra-class distance).
    Lower = tighter clusters.
    """
    centroids = compute_class_centroids(embeddings, labels)
    dists = []

    for i in range(len(labels)):
        c = labels[i]
        d = np.linalg.norm(embeddings[i] - centroids[c])
        dists.append(d)

    return float(np.mean(dists))


def compute_intra_inter_distances(embeddings, labels):
    centroids = compute_class_centroids(embeddings, labels)
    intra, inter = [], []

    classes = np.unique(labels)

    for i in range(len(labels)):
        true_c = labels[i]
        intra.append(
            np.linalg.norm(embeddings[i] - centroids[true_c])
        )

        other = [
            np.linalg.norm(embeddings[i] - centroids[c])
            for c in classes if c != true_c
        ]
        inter.append(min(other))

    return np.array(intra), np.array(inter)