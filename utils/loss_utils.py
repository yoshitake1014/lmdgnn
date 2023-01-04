import torch
from torch import nn


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

        self.BETA = 5

    def forward(self, x, y):
        loss = (( (x-y)*self.BETA )**2).mean()
        return loss
