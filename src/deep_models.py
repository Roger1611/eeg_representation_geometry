import torch
import torch.nn as nn


# ----------------------------
# EEGNet
# ----------------------------
class EEGNet(nn.Module):
    def __init__(self, chans, samples, classes=2):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (chans, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.second = nn.Sequential(
            nn.Conv2d(16, 16, (1, 16), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            feat = self.second(self.first(dummy))
            dim = feat.shape[1]

        self.classifier = nn.Linear(dim, classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.second(x)
        return self.classifier(x)


# ----------------------------
# DeepConvNet
# ----------------------------
class DeepConvNet(nn.Module):
    def __init__(self, chans, samples, classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5)),
            nn.Conv2d(25, 25, (chans, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 5)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),

            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            feat = self.features(dummy)
            dim = feat.shape[1]

        self.classifier = nn.Linear(dim, classes)

    def forward(self, x, return_embedding=False):
        x = x.unsqueeze(1)
        emb = self.features(x)
        logits = self.classifier(emb)

        if return_embedding:
            return logits, emb
        return logits


# ----------------------------
# ShallowConvNet
# ----------------------------
class ShallowConvNet(nn.Module):
    def __init__(self, chans, samples, classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 40, (1, 13)),
            nn.Conv2d(40, 40, (chans, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 35)),
            nn.Dropout(0.5),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            feat = self.features(dummy)
            dim = feat.shape[1]

        self.classifier = nn.Linear(dim, classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


# ----------------------------
# Projection Head Wrapper
# ----------------------------
class ProjectionHead(nn.Module):
    def __init__(self, backbone, embedding_dim, proj_dim=128):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        logits, emb = self.backbone(x, return_embedding=True)
        proj = self.projection(emb)
        return logits, emb, proj