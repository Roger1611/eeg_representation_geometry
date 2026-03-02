import torch
import torch.nn as nn


class DeepConvNet(nn.Module):
    def __init__(self, chans, samples, classes=4, dropout=0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(25, 25, (chans, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            z = self._forward_features(dummy)
            emb_dim = z.shape[1]

        self.classifier = nn.Linear(emb_dim, classes)

    def _forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.flatten(1)

    def forward(self, x, return_embedding=False):
        x = x.unsqueeze(1)
        z = self._forward_features(x)
        logits = self.classifier(z)

        if return_embedding:
            return logits, z
        return logits