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
        BATCH_SIZE = 50
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
        for i in range(TIME_STEP):
            G = nx.read_gpickle(f'datasets/as_733/month_{i+1}_graph.gpickle')
            graphs.append(G)
    elif dataset == 'hep':
        pass
    else:
        pass

    method = args.methods
    if method == 'lmdgnn':
        model = lmdgnn.LMDGNN(args,NUM_NODES,EMB_SIZE)

        for i in range(TIME_STEP-1):
            graph = graphs[i]
            next_graph = graphs[i+1]

            G_cen = nx.degree_centrality(graph)
            G_cen = sorted(G_cen.items(), key=lambda item:item[1], reverse=True)

            # Training
            sample_nodes = []
            sample_next_nodes = []
            for j in range(SAMPLE_SIZE):
                x = [0]*NUM_NODES
                y = [0]*NUM_NODES
                z = G_cen[j][0]
                neighbors_1 = graph.neighbors(z)
                neighbors_2 = next_graph.neighbors(z)
                for k in neighbors_1:
                    x[k-1] = 1
                sample_nodes.append(x)
                for k in neighbors_2:
                    y[k-1] = 1
                sample_next_nodes.append(y)

            X = torch.Tensor(sample_nodes)
            y = torch.Tensor(sample_next_nodes)
            train_dataset = TensorDataset(X,y)
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)

            lmdgnn.train(train_dataloader, model, loss_fn, optimizer)

            # Testing
            print('-'*50)
    elif method == 'dyngem':
        pass
    else:
        pass


if __name__ == "__main__":
    main()
