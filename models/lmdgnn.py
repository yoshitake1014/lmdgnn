import networkx as nx
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


class MLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lag_k = 100

        self.d_gate = nn.Linear(input_size+hidden_size, input_size)
        self.i_gate = nn.Linear(input_size+hidden_size, input_size)
        self.o_gate = nn.Linear(input_size+hidden_size, input_size)
        self.c_gate = nn.Linear(input_size+hidden_size, input_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x, hidden_state, memory_cell):
        if hidden_state is None:
            hidden_state = torch.zeros(1, self.hidden_size)
        if memory_cell is None:
            memory_cell = torch.zeros(1, self.hidden_size)

        cat = torch.cat((x, hidden_state), 1)
        d = self.d_gate(cat)
        d = self.sigmoid(d) * 0.5
        i = self.i_gate(cat)
        i = self.sigmoid(i)
        o = self.o_gate(cat)
        o = self.sigmoid(o)
        c_tilde = self.c_gate(cat)
        c_tilde = self.tanh(c_tilde)

        return


class MLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_state = None
        self.memory_cell = None
        self.mlstm_cell = MLSTMCell(self.input_size, self.hidden_size)


    def forward(self, x):
        x, self.hidden_state, self.memory_cell =\
        self.mlstm_cell(x, self.hidden_state, self.memory_cell)
        return x


class LMDGNN(nn.Module):
    def __init__(self, args, num_nodes, emb_size):
        super(LMDGNN, self).__init__()

        self.enc = Encoder(num_nodes, emb_size)
        self.dec = Decoder(emb_size, num_nodes)
        self.mlstm = MLSTM(emb_size, emb_size)


    def forward(self, x):
        x = self.enc(x)
        #x = self.mlstm(x)
        x = self.dec(x)
        return x


def train(dataloader, model, loss_fn, optimizer):
    THRESHOLD = 0.5

    model.train()

    for X,y in dataloader:
        X, y = X.to('cpu'), y.to('cpu')

        pred = model(X)
        threshold = torch.Tensor([THRESHOLD])
        pred = (pred >= threshold).type(torch.int)
        loss = loss_fn(pred, y)
        loss.requires_grad = True

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Loss: {loss}')


def test(dataloader, model, loss_fn):
    THRESHOLD = 0.5

    model.eval()

    correct = 0
    size = 0
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to('cpu'), y.to('cpu')

            pred = model(X)
            threshold = torch.Tensor([THRESHOLD])
            pred = (pred >= threshold).type(torch.int)
            correct += (pred == y).type(torch.float).sum()
            size += pred.numel()
    correct /= size

    print(f'Accuracy: {100*correct}')
