import networkx as nx
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self,
                input_size,
                output_size):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(input_size, 500),
                nn.ReLU(),
                nn.Linear(500,300),
                nn.ReLU(),
                nn.Linear(300,output_size),
                nn.ReLU()
                )


    def forward(self,x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self,
                input_size,
                output_size):
        super(Decoder,self).__init__()
        self.decoder = nn.Sequential(
                nn.Linear(input_size,300),
                nn.ReLU(),
                nn.Linear(300,500),
                nn.ReLU(),
                nn.Linear(500,output_size),
                nn.Sigmoid()
                )


    def forward(self,x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self,
                num_nodes,
                emb_size):
        super(Autoencoder,self).__init__()
        self.enc = Encoder(num_nodes,emb_size)
        self.dec = Decoder(emb_size,num_nodes)


    def forward(self,x):
        x = self.enc(x)
        x = self.dec(x)
        return x


class LMDGNN(nn.Module):
    def __init__(self,
                args,
                num_nodes,
                emb_size):
        super(LMDGNN,self).__init__()
        self.autoencoder = Autoencoder(num_nodes,emb_size)


    def forward(self,x):
        return self.autoencoder(x)


def train(dataloader, model, loss_fn, optimizer):
    THRESHOLD = 0.5

    model.train()
    for X,y in dataloader:
        X,y = X.to('cpu'), y.to('cpu')

        pred = model(X)
        threshold = torch.Tensor([THRESHOLD])
        pred = (pred >= threshold).int()
        loss = loss_fn(pred,y)
        loss.requires_grad = True

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'loss: {loss}')


def test(dataloader, model):
    THRESHOLD = 0.5

    model.eval()
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to('cpu'), y.to('cpu')

            pred = model(X)
            threshold = torch.Tensor([0.5])
            pred = (pred >= threshold).int()
            correct = (pred == y).type(torch.int).sum()
            accuracy = correct / pred.numel()
            print(f'Accuracy: {accuracy}')
