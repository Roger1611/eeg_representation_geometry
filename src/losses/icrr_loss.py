import torch


def icrr_loss(embeddings, labels):
    loss = 0.0
    count = 0

    for c in labels.unique():
        idx = labels == c
        if idx.sum() < 2:
            continue

        class_emb = embeddings[idx]
        centroid = class_emb.mean(dim=0, keepdim=True)
        loss += ((class_emb - centroid) ** 2).mean()
        count += 1

    return loss / (count + 1e-8)