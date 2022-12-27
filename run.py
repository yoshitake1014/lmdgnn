import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
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
        BATCH_SIZE = 500
        EMB_SIZE = 150
        EPOCH = 1
        HIDDEN_SIZE = [500, 300,]
        NUM_NODES = 7716
        LEARNING_RATE = 1e-3
        L2_NORM = 1e-2
        SAMPLE_SIZE = 500
        #TIME_STEP = 733
        TIME_STEP = 50

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
        model = lmdgnn.LMDGNN(args, NUM_NODES, HIDDEN_SIZE, EMB_SIZE)

        for i in range(1, TIME_STEP-1):
            graph_1 = graphs[i-1]
            graph_2 = graphs[i]
            graph_3 = graphs[i+1]

            G_cen = nx.degree_centrality(graph_1)
            G_cen = sorted(G_cen.items(), key=lambda item:item[1], reverse=True)

            nodes_1 = []
            nodes_2 = []
            nodes_3 = []
            for j in range(SAMPLE_SIZE):
                x = [0]*(NUM_NODES+1)
                y = [0]*(NUM_NODES+1)
                z = [0]*(NUM_NODES+1)
                node = G_cen[j][0]
                x[NUM_NODES] = node
                y[NUM_NODES] = node
                z[NUM_NODES] = node
                neighbors_1 = graph_1.neighbors(node)
                neighbors_2 = graph_2.neighbors(node)
                neighbors_3 = graph_3.neighbors(node)
                for k in neighbors_1:
                    x[k] = 1
                nodes_1.append(x)
                for k in neighbors_2:
                    y[k] = 1
                nodes_2.append(y)
                for k in neighbors_3:
                    z[k] = 1
                nodes_3.append(z)

            X = torch.Tensor(nodes_1)
            X = X.to(torch.int32)
            Y = torch.Tensor(nodes_2)
            Y = Y.to(torch.int32)
            Z = torch.Tensor(nodes_3)
            Z = Z.to(torch.int32)

            train_dataset = TensorDataset(X,Y)
            test_dataset = TensorDataset(Y,Z)

            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

            loss_fn = nn.MSELoss()
            #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
            optimizer = torch.optim.Adam(model.parameters())
            #optimizer = torch.optim.Adam(model.parameters(), weight_decay=L2_NORM)

            print(f'Timestep: {i}')
            lmdgnn.train(train_dataloader, model, loss_fn, optimizer, NUM_NODES)
            lmdgnn.test(test_dataloader, model, NUM_NODES)

    elif method == 'dyngem':
        model = dyngem.DynGEM(args, NUM_NODES, HIDDEN_SIZE, EMB_SIZE)

        for i in range(1, TIME_STEP-1):
            graph_1 = graphs[i-1]
            graph_2 = graphs[i]
            graph_3 = graphs[i+1]

            G_cen = nx.degree_centrality(graph_1)
            G_cen = sorted(G_cen.items(), key=lambda item:item[1], reverse=True)

            nodes_1 = []
            nodes_2 = []
            nodes_3 = []
            for j in range(SAMPLE_SIZE):
                x = [0]*(NUM_NODES)
                y = [0]*(NUM_NODES)
                z = [0]*(NUM_NODES)
                node = G_cen[j][0]
                neighbors_1 = graph_1.neighbors(node)
                neighbors_2 = graph_2.neighbors(node)
                neighbors_3 = graph_3.neighbors(node)
                for k in neighbors_1:
                    x[k] = 1
                nodes_1.append(x)
                for k in neighbors_2:
                    y[k] = 1
                nodes_2.append(y)
                for k in neighbors_3:
                    z[k] = 1
                nodes_3.append(z)

            X = torch.Tensor(nodes_1)
            Y = torch.Tensor(nodes_2)
            Z = torch.Tensor(nodes_3)

            train_dataset = TensorDataset(X,Y)
            test_dataset = TensorDataset(Y,Z)

            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters())

            print(f'Timestep: {i}')
            dyngem.train(train_dataloader, model, loss_fn, optimizer)
            dyngem.test(test_dataloader, model)
    else:
        pass


if __name__ == "__main__":
    main()
