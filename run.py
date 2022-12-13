import argparse
import os

import networkx as nx
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import lmdgnn, dynaernn, dynrnn, dynae, dyngem
from utils import loss_function as lf


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='as', help='')
parser.add_argument('--methods', type=str, default='lmdgnn', help='')
args = parser.parse_args()


def main():
    dataset = args.datasets
    if dataset == 'as':
        BATCH_SIZE = 1
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
        for i in range(TIME_STEP):
            G = nx.read_gpickle(f'datasets/as_733/month_{i+1}_graph.gpickle')
            graphs.append(G)
    elif dataset == 'hep':
        pass
    else:
        pass

    method = args.methods
    if method == 'lmdgnn':
        model = lmdgnn.LMDGNN(args, NUM_NODES, EMB_SIZE)

        for i in range(TIME_STEP-2):
            graph = graphs[i]
            next_graph = graphs[i+1]
            test_graph = graphs[i+2]

            G_cen = nx.degree_centrality(graph)
            G_cen = sorted(G_cen.items(), key=lambda item:item[1], reverse=True)

            # Training
            nodes = []
            next_nodes = []
            test_nodes = []
            for j in range(SAMPLE_SIZE):
                x = [0]*NUM_NODES
                y = [0]*NUM_NODES
                z = [0]*NUM_NODES
                node = G_cen[j][0]
                neighbors_1 = graph.neighbors(node)
                neighbors_2 = next_graph.neighbors(node)
                neighbors_3 = test_graph.neighbors(node)
                for k in neighbors_1:
                    x[k-1] = 1
                nodes.append(x)
                for k in neighbors_2:
                    y[k-1] = 1
                next_nodes.append(y)
                for k in neighbors_3:
                    z[k-1] = 1
                test_nodes.append(z)

            X = torch.Tensor(nodes)
            Y = torch.Tensor(next_nodes)
            Z = torch.Tensor(test_nodes)
            train_dataset = TensorDataset(X,Y)
            test_dataset = TensorDataset(Y,Z)
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

            #loss_fn = nn.MSELoss()
            loss_fn = lf.WeightedMSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

            lmdgnn.train(train_dataloader, model, loss_fn, optimizer)

            # Testing
            lmdgnn.test(test_dataloader, model, loss_fn)

            print('-'*50)
    elif method == 'dyngem':
        pass
    else:
        pass


if __name__ == "__main__":
    main()
