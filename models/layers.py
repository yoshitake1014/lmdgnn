import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 500),
                nn.ReLU(),
                nn.Linear(500, 300),
                nn.ReLU(),
                nn.Linear(300, output_size),
                nn.ReLU()
                )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
                nn.Linear(input_size, 300),
                nn.ReLU(),
                nn.Linear(300, 500),
                nn.ReLU(),
                nn.Linear(500, output_size),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.decoder(x)
