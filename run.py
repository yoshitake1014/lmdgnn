import argparse
import os

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import lmdgnn, dynaernn, dynrnn, dynae, dyngem
from utils import graph_utils


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='as', help='')
parser.add_argument('--methods', type=str, default='lmdgnn', help='')
args = parser.parse_args()


def main():
    dataset = args.datasets

    if dataset == 'as':
        EMB_SIZE = 150
        #EPOCH = 1
        HIDDEN_SIZE = [500, 300,]
        #NUM_NODES = 7716
        #LEARNING_RATE = 1e-3
        SAMPLE_SIZE = 2000
        #TIME_STEP = 733
        TIME_STEP = 50

        BATCH_SIZE = SAMPLE_SIZE
        NUM_NODES = SAMPLE_SIZE

        listdir = os.listdir('datasets/as_733')
        listdir.sort()

        graphs = []

        for i in range(TIME_STEP):
            G = nx.read_gpickle(f'datasets/as_733/month_{i+1}_graph.gpickle')
            graphs.append(G)

        G_cen = nx.degree_centrality(graphs[-1])
        G_cen = sorted(G_cen.items(), key=lambda item:item[1], reverse=True)

        node_l = []
        for i in range(SAMPLE_SIZE):
            node_l.append(G_cen[i][0])

        for i in range(TIME_STEP):
            graphs[i] = graph_utils.sample_graph_nodes(graphs[i], node_l)

    elif dataset == 'hep':
        pass
    else:
        pass

    method = args.methods

    if method == 'lmdgnn':
        model = lmdgnn.LMDGNN(NUM_NODES, HIDDEN_SIZE, EMB_SIZE)

        for i in range(1, TIME_STEP-1):
            graph_1 = graphs[i-1]
            graph_2 = graphs[i]
            graph_3 = graphs[i+1]

            nodes_1 = []
            nodes_2 = []
            nodes_3 = []
            for j in range(SAMPLE_SIZE):
                x = [0]*(NUM_NODES+1)
                y = [0]*(NUM_NODES+1)
                z = [0]*(NUM_NODES+1)
                x[NUM_NODES] = j
                y[NUM_NODES] = j
                z[NUM_NODES] = j
                neighbors_1 = graph_1.neighbors(j)
                neighbors_2 = graph_2.neighbors(j)
                neighbors_3 = graph_3.neighbors(j)
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
            #optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-2)

            print(f'Timestep: {i}')
            lmdgnn.train(train_dataloader, model, loss_fn, optimizer, NUM_NODES)
            lmdgnn.test(test_dataloader, model, NUM_NODES)

    elif method == 'dyngem':
        model = dyngem.DynGEM(NUM_NODES, HIDDEN_SIZE, EMB_SIZE)

        for i in range(1, TIME_STEP-1):
            graph_1 = graphs[i-1]
            graph_2 = graphs[i]
            graph_3 = graphs[i+1]

            nodes_1 = []
            nodes_2 = []
            nodes_3 = []
            for j in range(SAMPLE_SIZE):
                x = [0]*(NUM_NODES)
                y = [0]*(NUM_NODES)
                z = [0]*(NUM_NODES)
                neighbors_1 = graph_1.neighbors(j)
                neighbors_2 = graph_2.neighbors(j)
                neighbors_3 = graph_3.neighbors(j)
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

    elif method == 'dynae':
        model = dynae.DynAE(NUM_NODES, HIDDEN_SIZE, EMB_SIZE)
        lookback = model.lookback

        for i in range(1+lookback, TIME_STEP-1):
            graph_1 = []
            graph_2_1 = []
            for j in range(lookback+1):
                graph_1.append(graphs[i-j-1])
                graph_2_1.append(graphs[i-j])
            graph_2_2 = graphs[i]
            graph_3 = graphs[i+1]

            nodes_1 = [list() for _ in range(SAMPLE_SIZE)]
            nodes_2_1 = [list() for _ in range(SAMPLE_SIZE)]
            for j in range(1+lookback):
                for k in range(SAMPLE_SIZE):
                    x = [0]*(NUM_NODES)
                    y = [0]*(NUM_NODES)
                    neighbors_1 = graph_1[j].neighbors(j)
                    neighbors_2_1 = graph_2_1[j].neighbors(j)
                    for l in neighbors_1:
                        x[l] = 1
                    nodes_1[k].extend(x)
                    for l in neighbors_2_1:
                        y[l] = 1
                    nodes_2_1[k].extend(y)
            nodes_2_2 = []
            nodes_3 = []
            for j in range(SAMPLE_SIZE):
                x = [0]*(NUM_NODES)
                y = [0]*(NUM_NODES)
                neighbors_2_2 = graph_2_2.neighbors(j)
                neighbors_3 = graph_3.neighbors(j)
                for k in neighbors_2_2:
                    x[k] = 1
                nodes_2_2.append(x)
                for k in neighbors_3:
                    y[k] = 1
                nodes_3.append(y)

            X = torch.Tensor(nodes_1)
            Y = torch.Tensor(nodes_2_2)

            _X = torch.Tensor(nodes_2_1)
            _Y = torch.Tensor(nodes_3)

            train_dataset = TensorDataset(X,Y)
            test_dataset = TensorDataset(_X,_Y)

            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters())

            print(f'Timestep: {i}')
            dynae.train(train_dataloader, model, loss_fn, optimizer)
            dynae.test(test_dataloader, model)

    elif method == 'dynrnn':
        model = dynrnn.DynRNN(NUM_NODES, NUM_NODES)
        lookback = model.lookback

        for i in range(1+lookback, TIME_STEP-1):
            graph_1 = []
            graph_2_1 = []
            for j in range(lookback+1):
                graph_1.append(graphs[i-j-1])
                graph_2_1.append(graphs[i-j])
            graph_2_2 = graphs[i]
            graph_3 = graphs[i+1]

            nodes_1 = [list() for _ in range(SAMPLE_SIZE)]
            nodes_2_1 = [list() for _ in range(SAMPLE_SIZE)]
            for j in range(1+lookback):
                for k in range(SAMPLE_SIZE):
                    x = [0]*(NUM_NODES)
                    y = [0]*(NUM_NODES)
                    neighbors_1 = graph_1[j].neighbors(j)
                    neighbors_2_1 = graph_2_1[j].neighbors(j)
                    for l in neighbors_1:
                        x[l] = 1
                    nodes_1[k].extend(x)
                    for l in neighbors_2_1:
                        y[l] = 1
                    nodes_2_1[k].extend(y)
            nodes_2_2 = []
            nodes_3 = []
            for j in range(SAMPLE_SIZE):
                x = [0]*(NUM_NODES)
                y = [0]*(NUM_NODES)
                neighbors_2_2 = graph_2_2.neighbors(j)
                neighbors_3 = graph_3.neighbors(j)
                for k in neighbors_2_2:
                    x[k] = 1
                nodes_2_2.append(x)
                for k in neighbors_3:
                    y[k] = 1
                nodes_3.append(y)

            X = torch.Tensor(nodes_1)
            Y = torch.Tensor(nodes_2_2)

            _X = torch.Tensor(nodes_2_1)
            _Y = torch.Tensor(nodes_3)

            train_dataset = TensorDataset(X,Y)
            test_dataset = TensorDataset(_X,_Y)

            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters())

            print(f'Timestep: {i}')
            dynae.train(train_dataloader, model, loss_fn, optimizer)
            dynae.test(test_dataloader, model)

    else:
        pass


if __name__ == "__main__":
   main()
