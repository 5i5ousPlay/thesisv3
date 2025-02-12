import pickle
import os
import networkx as nx
from grakel import Graph
from grakel.kernels import LovaszTheta


def save_to_pickle(data, filename):
    """
    Saves a Python object to a pickle file.

    Args:
        data: Any Python object to save
        filename (str): Target filepath for the pickle file

    Returns:
        None
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")


def load_from_pickle(filename):
    """
    Loads a Python object from a pickle file.

    Args:
        filename (str): Source filepath of the pickle file

    Returns:
        The deserialized Python object
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Data loaded from {filename}")
    return data


def get_directories_with_min_files(root_dir, min_file_count=5):
    """
    Finds directories containing at least the specified minimum number of files.

    Args:
        root_dir (str): Root directory to start search
        min_file_count (int): Minimum number of files required (default: 5)

    Returns:
        list: Directory names meeting the minimum file count criterion
    """
    qualifying_directories = []
    for dirpath, _, filenames in os.walk(root_dir):
        file_count = len([name for name in filenames if os.path.isfile(os.path.join(dirpath, name))])
        if file_count > min_file_count:
            qualifying_directories.append(os.path.basename(dirpath))
    return qualifying_directories


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
