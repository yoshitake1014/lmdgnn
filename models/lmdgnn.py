from collections import deque
from math import gamma

import networkx as nx
from sklearn.metrics import roc_curve, roc_auc_score
import torch
from torch import nn

from models.layers import Encoder, Decoder


class LMDGNN(nn.Module):
    def __init__(self, args, num_nodes, hidden_size, emb_size):
        super(LMDGNN, self).__init__()

        self.args = args
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.emb_size = emb_size

        self.enc = Encoder(num_nodes, hidden_size, emb_size)
        self.mlstms = MLSTMs(emb_size, emb_size, num_nodes)
        self.dec = Decoder(emb_size, hidden_size, num_nodes)

    def forward(self, x):
        nodes_i = x[:, self.num_nodes]
        x = x[:, :self.num_nodes].to(torch.float32)

        x = self.enc(x)
        x = self.mlstms(x, nodes_i)
        x = self.dec(x)
        return x


class MLSTMs(nn.Module):
    def __init__(self, input_size, hidden_size, num_nodes):
        super(MLSTMs, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes

        self.mlstms = [MLSTM(input_size, hidden_size)]*num_nodes

    def forward(self, x, nodes_i):
        batch_size = x.size(0)

        res = torch.Tensor()
        for batch in range(batch_size):
            tmp = self.mlstms[nodes_i[batch].item()](x[batch])
            res = torch.cat([res, tmp], dim=0)
        res = torch.reshape(res, (-1, x.size(1)))

        return res


class MLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_state = None
        self.memory_cell = None
        self.mlstm_cell = MLSTMCell(self.input_size, self.hidden_size)

    def forward(self, x):
        self.hidden_state, self.memory_cell =\
        self.mlstm_cell(x, self.hidden_state, self.memory_cell)
        return self.hidden_state


class MLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lag_k = 50
        self.i_c_tilde = deque([None]*self.lag_k)

        self.d_u_gate = nn.Linear(hidden_size, input_size)
        self.d_w_gate = nn.Linear(input_size, input_size)
        self.i_u_gate = nn.Linear(hidden_size, input_size)
        self.i_w_gate = nn.Linear(input_size, input_size)
        self.o_u_gate = nn.Linear(hidden_size, input_size)
        self.o_w_gate = nn.Linear(input_size, input_size)
        self.c_u_gate = nn.Linear(hidden_size, input_size)
        self.c_w_gate = nn.Linear(input_size, input_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def compute_weights(self, d_i_gate):
        weights = [1.]*(self.lag_k)
        for k in range(self.lag_k):
            try:
                weights[k] = gamma(d_i_gate.item()+k) / (gamma(k+1)*gamma(d_i_gate.item()))
            except:
                pass
        weights = torch.tensor(weights)
        return weights

    def compute_memory_cell(self, d_gate):
        weights = torch.ones(self.lag_k,
                             self.hidden_size,
                             dtype=d_gate.dtype,
                             device=d_gate.device)
        for i in range(self.hidden_size):
            weights[:, i] = self.compute_weights(d_gate[i])

        memory_cell = []
        for i in  range(self.hidden_size):
            memory_cell_i = 0
            for k in range(self.lag_k):
                if self.i_c_tilde[k] is None:
                    break
                memory_cell_i += weights[k][i].item()*self.i_c_tilde[k][i].item()
            memory_cell.append(memory_cell_i)
        memory_cell = torch.tensor(memory_cell)

        return memory_cell

    def forward(self, x, hidden_state, memory_cell):
        if hidden_state is None:
            hidden_state = torch.zeros(self.hidden_size,
                                       dtype=x.dtype,
                                       device=x.device)
        if memory_cell is None:
            memory_cell = torch.zeros(self.hidden_size,
                                      dtype=x.dtype,
                                      device=x.device)

        d_gate = torch.add(self.d_u_gate(hidden_state), self.d_w_gate(x))
        d_gate = self.sigmoid(d_gate) * 0.5
        i_gate = torch.add(self.i_u_gate(hidden_state), self.i_w_gate(x))
        i_gate = self.sigmoid(i_gate)
        o_gate = torch.add(self.o_u_gate(hidden_state), self.o_w_gate(x))
        o_gate = self.sigmoid(o_gate)
        c_tilde = torch.add(self.c_u_gate(hidden_state), self.c_w_gate(x))
        c_tilde = self.tanh(c_tilde)

        self.i_c_tilde.pop()
        self.i_c_tilde.appendleft(torch.mul(i_gate, c_tilde))

        memory_cell = self.compute_memory_cell(d_gate)
        hidden_state = torch.mul(o_gate, self.tanh(memory_cell))

        return hidden_state, memory_cell


def train(dataloader, model, loss_fn, optimizer, num_nodes):
    size = len(dataloader.dataset)

    model.train()

    for i, (X,y) in enumerate(dataloader):
        y = y[:, :num_nodes].to(torch.float32)
        X, y = X.to('cpu'), y.to('cpu')

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (i+1)*len(X)
        print(f'Loss: {loss} [{current}/{size}]')


def test(dataloader, model, num_nodes):
    size = len(dataloader.dataset)

    model.eval()

    #y_true = np.ndarray(0)
    #y_score = np.ndarray(0)
    with torch.no_grad():
        for i, (X,y) in enumerate(dataloader):
            y = y[:, :num_nodes].to(torch.float32)
            X, y = X.to('cpu'), y.to('cpu').detach().numpy().copy().flatten()

            pred = model(X).to('cpu').detach().numpy().copy().flatten()

            #y_true = np.concatenate([y_true, y])
            #y_score = np.concatenate([y_score, pred])

            auc, current = roc_auc_score(y, pred), (i+1)*len(X)
            print(f'AUC: {auc} [{current}/{size}]')

    #auc = roc_auc_score(y_true, y_score)
    #print(f'AUC: {auc}')
