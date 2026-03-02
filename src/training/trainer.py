import torch
import torch.nn as nn
from src.losses.icrr_loss import icrr_loss


class Trainer:

    def __init__(self, model, optimizer, device="cuda", lambda_icrr=0.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_icrr = lambda_icrr
        self.ce_loss = nn.CrossEntropyLoss()

    def train_epoch(self, loader):
        self.model.train()

        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            self.optimizer.zero_grad()

            logits, z = self.model(xb, return_embedding=True)

            loss = self.ce_loss(logits, yb)

            if self.lambda_icrr > 0:
                loss += self.lambda_icrr * icrr_loss(z, yb)

            loss.backward()
            self.optimizer.step()