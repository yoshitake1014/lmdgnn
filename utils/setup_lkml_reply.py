import datetime

import networkx as nx


nodes = {}
node_id = 0

tmp = [[list() for _ in range(12)] for _ in range(8)]

with open('../datasets/out.lkml-reply') as f:
    for l in f:
        l = l.strip()
        if l.startswith('%'): continue
        node_i, node_j, weight, unix_time = l.split('\t')
        unix_time = datetime.datetime.fromtimestamp(int(unix_time))
        year = unix_time.year
        month = unix_time.month
        if node_i not in nodes:
            nodes[node_i] = node_id
            node_id += 1
        if node_j not in nodes:
            nodes[node_j] = node_id
            node_id += 1
        tmp[year-2006][month-1].append((nodes[node_i], nodes[node_j]))

file_no = 1
for i in range(8):
    for j in range(12):
        G = nx.DiGraph()
        G.add_nodes_from(range(27927))
        _tmp = tmp[i][j]
        for node_i, node_j in _tmp:
            G.add_edge(node_i, node_j, weight=1)
        file_name = f'../datasets/lkml_reply/month_{file_no}_graph.gpickle'
        nx.write_gpickle(G, file_name)
        file_no += 1
