import numpy as np


def compute_ess(embeddings, labels):
    ess = 0.0
    classes = np.unique(labels)

    for c in classes:
        idx = labels == c
        class_emb = embeddings[idx]
        centroid = class_emb.mean(axis=0)
        ess += np.mean(np.linalg.norm(class_emb - centroid, axis=1))

    return ess / len(classes)