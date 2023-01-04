import networkx as nx


def sample_graph_nodes(graph, node_l):
    node_l_inv = {v:k for k, v in enumerate(node_l)}
    sample_graph = nx.Graph()
    sample_graph.add_nodes_from(range(len(node_l)))
    for i, j, w in graph.edges(data='weight', default=1):
        try:
            v_i = node_l_inv[i]
            v_j = node_l_inv[j]
            sample_graph.add_edge(v_i, v_j, weight=w)
        except:
            continue
    return sample_graph
