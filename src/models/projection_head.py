import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.projection(x)