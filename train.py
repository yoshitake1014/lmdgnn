import argparse
import os

import networkx as nx
import pandas as pd

from models import lmdgnn


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='as', help='')
parser.add_argument('--methods', type=str, default='lmdgnn', help='')
args = parser.parse_args()


def main():
    dataset = args.datasets

    if dataset == 'as':
        NUM_NODES = 64602
        dir = os.listdir('datasets/as-733')
        dir.sort()
        graphs = []
        #for i in range(len(dir)):
        for i in range(3):
            G = nx.Graph()
            G.add_nodes_from(range(1,NUM_NODES+1))
            file = pd.read_csv(f'datasets/as-733/{dir[i]}',sep='\t',skiprows=4)
            file_ = [tuple(x) for x in file.values]
            G.add_edges_from(file_)
            graphs.append(G)
        pass

    method = args.methods

    if method == 'lmdgnn':
        model = lmdgnn.LMDGNN(args,graphs,NUM_NODES)
        pass

    pass


if __name__ == "__main__":
    main()
