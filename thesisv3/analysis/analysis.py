import networkx as nx
import numpy as np
import pandas as pd
from grakel import Graph
from grakel.kernels import WeisfeilerLehman
from networkx import Graph
from numpy.linalg import eigh
from scipy.sparse import csgraph
from sklearn.neighbors import kneighbors_graph


# ===============================
# Graph Analysis
# ===============================


def spectral_partition(distance_matrix):
    """
    Performs spectral partitioning on a distance matrix.

    Args:
        distance_matrix (np.array): Pairwise distance matrix

    Returns:
        tuple: Two arrays containing indices for the partitioned groups
    """
    sigma = np.mean(distance_matrix[np.nonzero(distance_matrix)])
    similarity_matrix = np.exp(-distance_matrix ** 2 / (2. * sigma ** 2))
    np.fill_diagonal(similarity_matrix, 0)

    laplacian = csgraph.laplacian(similarity_matrix, normed=True)
    eigenvalues, eigenvectors = eigh(laplacian)
    fiedler_vector = eigenvectors[:, 1]

    partition = fiedler_vector > 0
    return np.where(partition)[0], np.where(~partition)[0]


def kernighan_lin_partition(distance_matrix, seed=None):
    """
    Partition a set of nodes using the Kernighan-Lin algorithm based on a distance matrix.
    Returns two groups of node indices similar to the spectral_partition function.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        Matrix of distances between nodes
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    tuple: (group_1, group_2)
        Two arrays of node indices representing the partition
    """
    import numpy as np
    import networkx as nx

    # Create a graph from the distance matrix
    n = distance_matrix.shape[0]
    graph = nx.Graph()

    # Add all nodes
    for i in range(n):
        graph.add_node(i)

    # Add edges with weights based on inverse distance
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i, j] > 0:  # Only add edges for non-zero distances
                # Convert distance to weight (inverse of distance)
                weight = 1 / (distance_matrix[i, j] + 1e-5)  # Add small constant to avoid division by zero
                graph.add_edge(i, j, weight=weight)

    # Set random seed
    rng = np.random.default_rng(seed)

    # Use Kernighan-Lin bisection
    partition = nx.algorithms.community.kernighan_lin_bisection(graph, weight="weight", seed=rng)

    # Extract nodes from the partitions
    group_1 = np.array(list(partition[0]))
    group_2 = np.array(list(partition[1]))

    return group_1, group_2


def get_sub_distance_matrix(distance_matrix, group_indices):
    """
    Extracts a sub-matrix from a distance matrix based on group indices.

    Args:
        distance_matrix (np.array): Original distance matrix
        group_indices (np.array): Indices to extract

    Returns:
        np.array: Sub-matrix containing only the specified indices
    """
    return distance_matrix[np.ix_(group_indices, group_indices)]


def dist_mat_to_graph(k: int, distance_matrix):
    """
    Converts a distance matrix to a k-NN graph.

    Args:
        k (int): Number of nearest neighbors
        distance_matrix (np.array): Pairwise distance matrix

    Returns:
        networkx.Graph: K-nearest neighbors graph
    """
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
    return nx.from_scipy_sparse_array(knn_graph)

