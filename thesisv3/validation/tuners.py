import grakel
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from scipy.stats import shapiro
from scipy.stats import ttest_ind, mannwhitneyu
from statistics import mean
from networkx.algorithms.community import kernighan_lin_bisection
from thesisv3.utils.helpers import compare_graphs_kernel
import matplotlib.pyplot as plt


class KNNGraphTuner:
    """
    Tuner class for calculating the p-values for within and between graph scores
    for each corresponding k values within a specified range.
    """
    distance_matrices = None
    graph_kernel = None
    seed = None
    min_k = None
    max_k = None
    k_step = None

    def __init__(self, distance_matrices: list[np.ndarray], graph_kernel: grakel.kernels.Kernel, seed=42,
                 min_k=1, max_k=10, k_step=1):
        """
        Initializes a Tuner instance.
        Args:
            distance_matrices (list): List of numpy arrays (2D distance matrices)
            graph_kernel (grakel.kernels.Kernel): Graph kernel used for calculating within and between graph scores
            seed (int): RNG seed for reproducibility
            min_k (int): minimum k value to test
            max_k (int): maximum k value to test
            k_step (int): interval step for k value testing
        """
        self.distance_matrices = distance_matrices
        self.graph_kernel = graph_kernel
        self.seed = seed
        self.min_k = min_k
        self.max_k = max_k
        self.k_step = k_step

    def _construct_graph(self, k: int, distance_matrix: np.ndarray):
        knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
        G = nx.from_scipy_sparse_array(knn_graph)

        G = self._ensure_connectivity(G, distance_matrix)

        return G

    def _ensure_connectivity(self, G: nx.Graph, distance_matrix: np.ndarray):
        if not nx.is_connected(G):

            components = list(nx.connected_components(G))

            for i in range(len(components) - 1):
                min_dist = np.inf
                closest_pair = None
                for node1 in components[i]:
                    for node2 in components[i + 1]:
                        dist = distance_matrix[node1, node2]
                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = (node1, node2)

                G.add_edge(closest_pair[0], closest_pair[1])

        return G

    def _partition(self, graph: nx.Graph, distance_matrix: np.ndarray):
        rng = np.random.default_rng(self.seed)

        # Convert unweighted graph to weighted using distance matrix
        for u, v in graph.edges():
            graph[u][v]['weight'] = 1 / (distance_matrix[u, v] + 1e-5)  # Add a small constant to avoid division by zero

        # Use Kernighan-Lin bisection to split the graph into two partitions
        partition = kernighan_lin_bisection(graph, weight="weight", seed=rng)

        # Extract nodes from the partitions
        group1 = list(partition[0])
        group2 = list(partition[1])

        # Create subgraphs
        subgraph1 = graph.subgraph(group1).copy()
        subgraph2 = graph.subgraph(group2).copy()

        # Comment this in if you want some subgraph samples

        # plt.figure(figsize=(8, 6))
        # nx.draw(subgraph1, with_labels=True, node_color='lightblue', edge_color='gray')
        # plt.show()
        #
        # plt.figure(figsize=(8, 6))
        # nx.draw(subgraph2, with_labels=True, node_color='lightgreen', edge_color='gray')
        # plt.show()

        return subgraph1, subgraph2

    def _kernel_based_similarity(self, k: int):
        within_graph_scores = []
        between_graph_scores = []

        graphs = []
        for distance_matrix in self.distance_matrices:
            graph = self._construct_graph(distance_matrix=distance_matrix, k=k)
            partition1, partition2 = self._partition(graph=graph, distance_matrix=distance_matrix)
            graphs.append(
                {
                    "split_graphs": (partition1, partition2),
                    "whole_graph": graph
                    }
                )
            similarity_within = compare_graphs_kernel([partition1, partition2], self.graph_kernel)[0, 1]
            within_graph_scores.append(similarity_within)

        for i in range(len(graphs)):
            for j in range(i+1, len(graphs)):
                whole_graph_1 = graphs[i]['whole_graph']
                whole_graph_2 = graphs[j]['whole_graph']

                similarity_between = compare_graphs_kernel([whole_graph_1, whole_graph_2], self.graph_kernel)[0, 1]
                between_graph_scores.append(similarity_between)

        return within_graph_scores, between_graph_scores

    def calculate_graph_statistics(self) -> pd.DataFrame:
        """
        Calculates the normality of the within and between graph scores as well as the
        corresponding p-value for each k-value within the specified range.

        Returns:
            graph_statistics (pd.DataFrame): Dataframe containing the parametric and non-parametric
            p-values for each k-value within the specified range.
        """
        graph_statistics = pd.DataFrame(columns=['k', 'normality_wtihin', 'normality_between', 'average_within',
                                                 'average_between', 'parametric_p_value', 'non_parametric_p_value'])
        for k in range(self.min_k, self.max_k, self.k_step):
            idx = len(graph_statistics)
            graph_statistics.loc[idx, 'k'] = k
            print(f"Calculating graph statistics at k = {k}")
            within_graph_scores, between_graph_scores = self._kernel_based_similarity(k)

            # Normality tests
            stat, p_value_within = shapiro(within_graph_scores)
            stat, p_value_between = shapiro(between_graph_scores)

            graph_statistics.loc[idx, 'normality_wtihin'] = p_value_within
            graph_statistics.loc[idx, 'normality_between'] = p_value_between

            # Average Scores
            graph_statistics.loc[idx, 'average_within'] = mean(within_graph_scores)
            graph_statistics.loc[idx, 'average_between'] = mean(between_graph_scores)

            # Parametric test
            t_stat, p_value = ttest_ind(within_graph_scores, between_graph_scores)
            graph_statistics.loc[idx,'parametric_p_value'] = p_value

            # Non-parametric test
            u_stat, p_value_non_parametric = mannwhitneyu(within_graph_scores, between_graph_scores)
            graph_statistics.loc[idx,'non_parametric_p_value'] = p_value_non_parametric

        return graph_statistics

    def calculate_and_graph(self) -> pd.DataFrame:
        """
        Calculates the normality of the within and between graph scores as well as the
        corresponding p-value for each k-value within the specified range. Generates
        a graph comparing p-values against varying k-values relative to the 0.05
        significance threshold.

        Returns:
            graph_statistics (pd.DataFrame): Dataframe containing the parametric and non-parametric
            p-values for each k-value within the specified range.
        """
        graph_statistics = self.calculate_graph_statistics()
        plt.figure(figsize=(10, 6))

        plt.plot(graph_statistics['k'], graph_statistics['parametric_p_value'], label='Parametric P-Value',
                 marker='o', linestyle='-', color='blue')
        plt.plot(graph_statistics['k'], graph_statistics['non_parametric_p_value'], label='Non-Parametric P-Value',
                 marker='s', linestyle='--', color='orange')

        plt.xlabel('k Values')
        plt.ylabel('P-Values')
        plt.title(f'P-Values vs. k ({self.graph_kernel.__class__.__name__})')
        plt.axhline(y=0.05, color='red', linestyle=':', label='Significance Threshold (p=0.05)')
        plt.legend(loc='best')

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        return graph_statistics
