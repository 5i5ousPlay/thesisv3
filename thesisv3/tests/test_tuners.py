import unittest
import numpy as np
import networkx as nx
from grakel.kernels import WeisfeilerLehman
from thesisv3.validation import tuners


class TestKNNGraphTuner(unittest.TestCase):
    def setUp(self):
        dist_mats = np.load('thesisv3/tests/distance_matrices.npz')
        self.distance_matrices = []
        for i in range(len(dist_mats)):
            self.distance_matrices.append(dist_mats[f'arr_{i}'])

        self.kernel = WeisfeilerLehman(normalize=True)

        self.tuner = tuners.KNNGraphTuner(
            distance_matrices=self.distance_matrices,
            graph_kernel=self.kernel,
            seed=42,
            min_k=1,
            max_k=6,
            k_step=1
            )

    def test_graph_construction(self):
        k = 5
        G = self.tuner._construct_graph(k, self.distance_matrices[0])
        self.assertTrue(nx.is_connected(G), "Graph should be connected after ensuring connectivity.")
        self.assertEqual(len(G.nodes), 37, "Graph should have the correct number of nodes.")

    def test_partitioning(self):
        k = 5
        G = self.tuner._construct_graph(k, self.distance_matrices[0])
        subgraph1, subgraph2 = self.tuner._partition(G, self.distance_matrices[0])
        self.assertGreater(len(subgraph1), 0, "Partition 1 should contain nodes.")
        self.assertGreater(len(subgraph2), 0, "Partition 2 should contain nodes.")
        self.assertEqual(len(subgraph1) + len(subgraph2), len(G.nodes), "Partitions should cover all nodes.")

    def test_similarity_calculation(self):
        k = 5
        within_scores, between_scores = self.tuner._kernel_based_similarity(k)
        self.assertGreater(len(within_scores), 0, "Within graph scores should not be empty.")
        self.assertGreater(len(between_scores), 0, "Between graph scores should not be empty.")

    def test_calculate_graph_statistics(self):
        graph_stats = self.tuner.calculate_graph_statistics()
        self.assertFalse(graph_stats.empty, "Graph statistics dataframe should not be empty.")
        self.assertIn('k', graph_stats.columns,
                      "'k' column should be present in the graph statistics dataframe.")
        self.assertIn('parametric_p_value', graph_stats.columns,
                      "'parametric_p_value' column should be present.")

    def test_calculate_and_graph(self):
        try:
            graph_stats = self.tuner.calculate_and_graph()
            self.assertFalse(graph_stats.empty, "Graph statistics dataframe should not be empty after graphing.")
        except Exception as e:
            self.fail(f"calculate_and_graph raised an exception: {e}")
