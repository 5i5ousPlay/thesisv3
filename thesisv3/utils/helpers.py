import networkx as nx
from grakel import Graph
from grakel.kernels import LovaszTheta


def nx_to_grakel(G):
    edges = list(G.edges())

    # Create dummy node labels if none exist
    if not nx.get_node_attributes(G, 'label'):
        labels = {node: idx for idx, node in enumerate(G.nodes())}
    else:
        labels = nx.get_node_attributes(G, 'label')

    # Ensure the graph is formatted correctly for Grakel
    return Graph(edges, node_labels=labels)


def compare_graphs_kernel(graph_list: list, graph_kernel):
    grakel_graphs = [nx_to_grakel(g) for g in graph_list]
    kernel = graph_kernel
    if isinstance(kernel, LovaszTheta):
        max_dim = max(len(g.nodes()) for g in graph_list)
        similarity_matrix = LovaszTheta(normalize=True, max_dim=max_dim).fit_transform(grakel_graphs)
        return similarity_matrix
    similarity_matrix = kernel.fit_transform(grakel_graphs)
    return similarity_matrix
