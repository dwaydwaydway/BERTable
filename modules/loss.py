import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def pairwise_distances(self, x, y):
        x_norm = (x**2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def forward(self, logit, candidates, target):
        scores = self.pairwise_distances(logit, candidates)
        positive = scores.gather(1, target).expand_as(scores)
        loss = (self.margin + positive - scores).clamp(min=0)
        return loss.mean(1)