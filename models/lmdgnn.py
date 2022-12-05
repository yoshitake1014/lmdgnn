import torch
from torch import nn
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU()
                )

    def forward(self,x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.decoder = nn.Sequential(
                nn.Linear(64,128),
                nn.ReLU(),
                nn,Linear(128,512),
                nn.ReLU(),
                nn,Linear(512,28*28),
                nn.ReLU()
                )

    def forward(self,x):
        return self.decoder(x)


class LMDGNN():
    def __init__(self,args):
        pass
