from sklearn import metrics
import torch
from torch import nn

from models.layers import LSTM


class DynRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DynRNN, self).__init__()
        self.lookback = 2

        self.lstm = LSTM(input_size*(1+self.lookback), hidden_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        res = torch.Tensor()
        for batch in range(batch_size):
            tmp = self.lstm(x[batch])
            res = torch.cat([res, tmp], dim=0)
        res = torch.reshape(res, (-1, x.size(0)))
        return res


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()

    for i, (X,y) in enumerate(dataloader):
        X, y = X.to('cpu'), y.to('cpu')

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (i+1)*len(X)
        print(f'Loss: {loss} [{current}/{size}]')


def test(dataloader, model):
    size = len(dataloader.dataset)

    model.eval()

    with torch.no_grad():
        for i, (X,y) in enumerate(dataloader):
            X, y = X.to('cpu'), y.to('cpu').detach().numpy().copy().flatten()

            pred = model(X).to('cpu').detach().numpy().copy().flatten()

            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            auc = metrics.auc(fpr, tpr)

            precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
            prauc = metrics.auc(recall, precision)

            current = (i+1)*len(X)

            print(f'AUC: {auc} [{current}/{size}]')
            print(f'PRAUC: {prauc} [{current}/{size}]')
