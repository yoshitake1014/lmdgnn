import argparse
import os

import networkx as nx
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import lmdgnn, dynaernn, dynrnn, dynae, dyngem


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='as', help='')
parser.add_argument('--methods', type=str, default='lmdgnn', help='')
args = parser.parse_args()


def main():
    dataset = args.datasets
    if dataset == 'as':
        EMB_SIZE = 150
        NUM_NODES = 7716
        LEARNING_RATE = 1e-3
        #SAMPLE_SIZE = 2000
        SAMPLE_SIZE = 50
        #TIME_STEP = 733
        TIME_STEP = 5

        listdir = os.listdir('datasets/as_733')
        listdir.sort()
        graphs = []
        #for i in range(len(listdir)):
        for i in range(5):
            G = nx.read_gpickle(f'datasets/as_733/month_{i+1}_graph.gpickle')
            graphs.append(G)

    elif dataset == 'hep':
        pass
    else:
        pass

    method = args.methods
    if method == 'lmdgnn':
        model = lmdgnn.LMDGNN(args,NUM_NODES,EMB_SIZE)

        def train(dataloader, model, loss_fn, optimizer):
            pass

        def test(dataloader, model):
            pass

        for i in range(TIME_STEP-1):
            graph = graphs[i]
            target_graph = graphs[i+1]

            G_cen = nx.degree_centrality(graph)
            G_cen = sorted(G_cen.items(), key=lambda item:item[1], reverse=True)

            sample_nodes = []
            target_nodes = []
            for j in range(SAMPLE_SIZE):
                x = [0]*7716
                y = [0]*7716
                node = G_cen[j][0]
                neighbors = graph.neighbors(node)
                target_neighbors = target_graph.neighbors(node)
                for k in neighbors:
                    x[k-1] = 1
                sample_nodes.append(x)
                for k in target_neighbors:
                    y[k-1] = 1
                target_nodes.append(y)

            X = torch.Tensor(sample_nodes)
            y = torch.Tensor(target_nodes)
            train_dataset = TensorDataset(X,y)

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)

    elif method == 'dyngem':
        pass
    else:
        pass


if __name__ == "__main__":
    main()
