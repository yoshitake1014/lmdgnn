import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size[0]),
                nn.ReLU(),
                nn.Linear(hidden_size[0], hidden_size[1]),
                nn.ReLU(),
                nn.Linear(hidden_size[1], output_size),
                nn.ReLU()
                )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
                nn.Linear(input_size, hidden_size[1]),
                nn.ReLU(),
                nn.Linear(hidden_size[1], hidden_size[0]),
                nn.ReLU(),
                nn.Linear(hidden_size[0], output_size),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.decoder(x)
