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


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_state = None
        self.memory_cell = None
        self.lstm_cell = LSTMCell(self.input_size, self.hidden_size)

    def forward(self, x):
        self.hidden_state, self.memory_cell =\
        self.lstm_cell(x, self.hidden_state, self.memory_cell)
        return self.hidden_state


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.f_u_gate = nn.Linear(hidden_size, hidden_size)
        self.f_w_gate = nn.Linear(input_size, hidden_size)
        self.i_u_gate = nn.Linear(hidden_size, hidden_size)
        self.i_w_gate = nn.Linear(input_size, hidden_size)
        self.o_u_gate = nn.Linear(hidden_size, hidden_size)
        self.o_w_gate = nn.Linear(input_size, hidden_size)
        self.c_u_gate = nn.Linear(hidden_size, hidden_size)
        self.c_w_gate = nn.Linear(input_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden_state, memory_cell):
        if hidden_state is None:
            hidden_state = torch.zeros(self.hidden_size,
                                       dtype=x.dtype,
                                       device=x.device)
        if memory_cell is None:
            memory_cell = torch.zeros(self.hidden_size,
                                      dtype=x.dtype,
                                      device=x.device)

        f_gate = torch.add(self.f_u_gate(hidden_state), self.f_w_gate(x))
        f_gate = self.sigmoid(f_gate)
        i_gate = torch.add(self.i_u_gate(hidden_state), self.i_w_gate(x))
        i_gate = self.sigmoid(i_gate)
        o_gate = torch.add(self.o_u_gate(hidden_state), self.o_w_gate(x))
        o_gate = self.sigmoid(o_gate)
        c_tilde = torch.add(self.c_u_gate(hidden_state), self.c_w_gate(x))
        c_tilde = self.tanh(c_tilde)

        first = torch.mul(f_gate, memory_cell)
        second = torch.mul(i_gate, c_tilde)
        memory_cell = torch.add(first, second)
        hidden_state = torch.mul(o_gate, self.tanh(memory_cell))

        return hidden_state, memory_cell
