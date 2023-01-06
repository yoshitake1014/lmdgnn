import os

import networkx as nx


routers = {}
router_id = 0
listdir = sorted(os.listdir('../datasets/as-caida'))
file_no = 1

for i in listdir:
    with open(f'../datasets/as-caida/{i}') as f:
        G = nx.DiGraph()
        G.add_nodes_from(range(31379))
        for l in f:
            l = l.strip()
            if l.startswith('#'): continue
            node_i,node_j, _ = l.split('\t')
            if node_i not in routers:
                routers[node_i] = router_id
                router_id += 1
            if node_j not in routers:
                routers[node_j] = router_id
                router_id += 1
            G.add_edge(routers[node_i], routers[node_j], weight=1)
        file_name = f'../datasets/as_caida/month_{file_no}_graph.gpickle'
        nx.write_gpickle(G,file_name)
        file_no += 1
