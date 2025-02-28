import os
import pickle
import traceback

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
from thesisv3.classism import GraphBuilder, MusicSegmentAnalyzer, MusicFileManager
import matplotlib.pyplot as plt
import grakel as gk


class KNNGraphTuner:
    """
    Tuner class for calculating the p-values for within and between graph scores
    for each corresponding k values within a specified range.
    """

    def __init__(self,
                 graph_kernel: grakel.kernels.Kernel,
                 seed=42,
                 min_k=1,
                 max_k=10,
                 k_step=1,
                 output_dir='./tuner_output',
                 batcher_output_dir=None):
        """
        Initializes a Tuner instance.
        Args:
            graph_kernel (grakel.kernels.Kernel): Graph kernel used for calculating within and between graph scores
            seed (int): RNG seed for reproducibility
            min_k (int): minimum k value to test
            max_k (int): maximum k value to test
            k_step (int): interval step for k value testing
            output_dir (str): directory to save progress and results
            batcher_output_dir (str, optional): path to the GraphBatcher output directory to use instead of processing files again
        """
        self.distance_matrices = []
        self.segments = []
        self.graph_kernel = graph_kernel
        self.seed = seed
        self.min_k = min_k
        self.max_k = max_k
        self.k_step = k_step
        self.file_manager = MusicFileManager()
        self.analyzer = MusicSegmentAnalyzer()

        # Track processed files
        self.processed_files = []

        # Create dictionaries for mapping files to their data
        self.segment_dict = {}
        self.distmat_dict = {}

        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Define paths for each pickle file
        self.segments_path = os.path.join(self.output_dir, 'tuner_segments.pkl')
        self.distmat_path = os.path.join(self.output_dir, 'tuner_distance_matrices.pkl')
        self.processed_files_path = os.path.join(self.output_dir, 'tuner_processed_files.pkl')
        self.results_path = os.path.join(self.output_dir, 'tuner_results.pkl')

        # Dictionary to store results for each k value
        self.results = {}

        # Flag to track if we're using batcher data
        self.using_batcher_data = batcher_output_dir is not None

        # Load previous progress or batcher data
        if self.using_batcher_data:
            self.load_from_batcher(batcher_output_dir)
        else:
            self.load_progress()

    def _calculate_segments_and_distance_matrix(self) -> None:
        """
        Analyze files to generate segments and distance matrices, with progress tracking.
        """
        # If we're using batcher data, skip the file processing
        if self.using_batcher_data and self.processed_files:
            print(f"Using pre-processed data from GraphBatcher ({len(self.processed_files)} files)")
            return

        for file in self.file_manager.files:
            # Skip files we've already processed
            if file in self.processed_files:
                print(f"Skipping already processed file: {file}")
                continue

            print(f"Analyzing {file}")
            try:
                self.analyzer.run(self.file_manager.files[file])
                if not np.isnan(self.analyzer.distance_matrix).any():
                    # Add to lists
                    self.segments.append(self.analyzer.prepped_segments)
                    self.distance_matrices.append(self.analyzer.distance_matrix)
                    self.processed_files.append(file)

                    # Add to dictionaries with file as key
                    self.segment_dict[file] = self.analyzer.prepped_segments
                    self.distmat_dict[file] = self.analyzer.distance_matrix

                    # Save progress after each file
                    self.save_progress()
                else:
                    print(f"Distance Matrix of {file} has null values and cannot be converted into a graph. Skipping")
            except Exception as e:
                print(f"Error parsing: {file} at {self.file_manager.files[file]}. Skipping file")
                print(traceback.format_exc())
                continue

    def _partition(self, graph: nx.Graph, distance_matrix: np.ndarray):
        # Existing partition method unchanged
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

        return subgraph1, subgraph2

    def _kernel_based_similarity(self, k: int):
        # Existing kernel similarity method unchanged
        within_graph_scores = []
        between_graph_scores = []

        whole_graphs = []
        for distance_matrix, segments in zip(self.distance_matrices, self.segments):
            builder = GraphBuilder(k=k, distance_matrix=distance_matrix, segments=segments)
            graph = builder.construct_graph()
            whole_graphs.append(graph)

        distance_matrices = self.distance_matrices

        for graph, distance_matrix in zip(whole_graphs, distance_matrices):
            partition1, partition2 = self._partition(graph=graph, distance_matrix=distance_matrix)
            similarity_within = compare_graphs_kernel([partition1, partition2], self.graph_kernel)[0, 1]
            within_graph_scores.append(similarity_within)

        for i in range(len(whole_graphs)):
            for j in range(i + 1, len(whole_graphs)):
                whole_graph_1 = whole_graphs[i]
                whole_graph_2 = whole_graphs[j]

                similarity_between = compare_graphs_kernel([whole_graph_1, whole_graph_2], self.graph_kernel)[0, 1]
                between_graph_scores.append(similarity_between)

        return within_graph_scores, between_graph_scores

    def calculate_graph_statistics(self) -> pd.DataFrame:
        """
        Calculates the normality of the within and between graph scores as well as the
        corresponding p-value for each k-value within the specified range, with
        progress tracking and saving.

        Returns:
            graph_statistics (pd.DataFrame): Dataframe containing the parametric and non-parametric
            p-values for each k-value within the specified range.
        """
        if not self.distance_matrices or not self.segments:
            self._calculate_segments_and_distance_matrix()

        graph_statistics = pd.DataFrame(columns=['k', 'normality_wtihin', 'normality_between', 'average_within',
                                                 'average_between', 'parametric_p_value', 'non_parametric_p_value'])

        # Check if we have cached results for any k values
        for k in range(self.min_k, self.max_k, self.k_step):
            if k in self.results:
                print(f"Using cached results for k = {k}")
                idx = len(graph_statistics)
                result = self.results[k]
                graph_statistics.loc[idx] = result
                continue

            idx = len(graph_statistics)
            print(f"Calculating graph statistics at k = {k}")
            within_graph_scores, between_graph_scores = self._kernel_based_similarity(k)

            # Normality tests
            stat, p_value_within = shapiro(within_graph_scores)
            stat, p_value_between = shapiro(between_graph_scores)

            # Average Scores
            avg_within = mean(within_graph_scores)
            avg_between = mean(between_graph_scores)

            # Parametric test
            t_stat, p_value = ttest_ind(within_graph_scores, between_graph_scores)

            # Non-parametric test
            u_stat, p_value_non_parametric = mannwhitneyu(within_graph_scores, between_graph_scores)

            # Store the results
            result = [k, p_value_within, p_value_between, avg_within, avg_between, p_value, p_value_non_parametric]
            graph_statistics.loc[idx] = result

            # Cache the results
            self.results[k] = result

            # Save results after each k value
            self.save_progress()

        return graph_statistics

    def calculate_and_graph(self) -> pd.DataFrame:
        """
        Calculates the statistics and generates a plot.
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

        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'p_values_vs_k.png'))
        plt.show()

        return graph_statistics

    def save_progress(self):
        """Save each variable to its own pickle file, overwriting previous versions."""

        # Helper function for safe saving
        def safe_save(data, path):
            temp_path = f"{path}.tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
            if os.path.exists(path):
                os.remove(path)
            os.rename(temp_path, path)

        # Save each dictionary to its own file
        safe_save(self.segment_dict, self.segments_path)
        safe_save(self.distmat_dict, self.distmat_path)
        safe_save(self.processed_files, self.processed_files_path)
        safe_save(self.results, self.results_path)

        print(f"Progress saved: {len(self.processed_files)} files processed, {len(self.results)} k values calculated")

    def load_progress(self):
        """Load previous progress from individual pickle files if available."""

        # Helper function for safe loading
        def safe_load(path, default_value):
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            return default_value

        # Load each dictionary from its own file
        self.segment_dict = safe_load(self.segments_path, {})
        self.distmat_dict = safe_load(self.distmat_path, {})
        self.processed_files = safe_load(self.processed_files_path, [])
        self.results = safe_load(self.results_path, {})

        # Rebuild the lists from the dictionaries
        if self.processed_files:
            self.segments = [self.segment_dict[f] for f in self.processed_files]
            self.distance_matrices = [self.distmat_dict[f] for f in self.processed_files]

            print(f"Loaded previous progress: {len(self.processed_files)} files already processed, "
                  f"{len(self.results)} k values already calculated")

    def load_from_batcher(self, batcher_output_dir):
        """Load data from a GraphBatcher output directory."""
        print(f"Attempting to load data from GraphBatcher output directory: {batcher_output_dir}")

        # Define batcher paths
        batcher_segments_path = os.path.join(batcher_output_dir, 'segments.pkl')
        batcher_distmat_path = os.path.join(batcher_output_dir, 'distance_matrices.pkl')
        batcher_processed_files_path = os.path.join(batcher_output_dir, 'processed_files.pkl')

        # Helper function for safe loading
        def safe_load(path, default_value):
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    return default_value
            else:
                print(f"File not found: {path}")
                return default_value

        # Load batcher data
        batcher_segment_dict = safe_load(batcher_segments_path, {})
        batcher_distmat_dict = safe_load(batcher_distmat_path, {})
        batcher_processed_files = safe_load(batcher_processed_files_path, [])

        if not batcher_processed_files:
            print("No data found in the batcher output directory. Will process files directly.")
            return

        # Import batcher data into tuner
        self.segment_dict = batcher_segment_dict
        self.distmat_dict = batcher_distmat_dict
        self.processed_files = batcher_processed_files

        # Load tuner-specific results if they exist
        self.results = safe_load(self.results_path, {})

        # Rebuild the lists from the dictionaries
        self.segments = [self.segment_dict[f] for f in self.processed_files]
        self.distance_matrices = [self.distmat_dict[f] for f in self.processed_files]

        print(f"Successfully loaded data from GraphBatcher: {len(self.processed_files)} files")
        print(f"Previously calculated results for {len(self.results)} k values")


def compare_kernels(batcher_output_dir='./batcher_output', min_k=1, max_k=10, k_step=1):
    """
    Compare multiple graph kernels using the same dataset.

    Args:
        batcher_output_dir (str): Path to the GraphBatcher output directory
        min_k (int): Minimum k value to test
        max_k (int): Maximum k value to test
        k_step (int): Step size for k values

    Returns:
        dict: Dictionary mapping kernel names to their results DataFrames
    """
    # Define kernels to test
    kernels = {
        'WeisfeilerLehman': gk.WeisfeilerLehman(normalize=True),
        # 'PyramidMatch': gk.PyramidMatch(normalize=True),
        'ShortestPath': gk.ShortestPath(normalize=True),
        'GraphletSampling': gk.GraphletSampling(normalize=True),
        'RandomWalkLabeled': gk.RandomWalkLabeled(),
        'WeisfeilerLehman (non-normalized)': gk.WeisfeilerLehman(normalize=False),
        # 'PyramidMatch (non-normalized)': gk.PyramidMatch(normalize=False),
        'ShortestPath (non-normalized)': gk.ShortestPath(normalize=False),
        'GraphletSampling (non-normalized)': gk.GraphletSampling(normalize=False),
    }

    results = {}
    figures = []

    # Loop through each kernel
    for name, kernel in kernels.items():
        print(f"\n\n{'=' * 50}")
        print(f"Testing kernel: {name}")
        print(f"{'=' * 50}\n")

        # Create tuner with this kernel
        tuner = KNNGraphTuner(
            graph_kernel=kernel,
            batcher_output_dir=batcher_output_dir,
            min_k=min_k,
            max_k=max_k,
            k_step=k_step,
            output_dir=f'./tuner_output_{name.lower()}'  # Separate output dir for each kernel
        )

        # Run analysis
        result_df = tuner.calculate_graph_statistics()
        results[name] = result_df

        # Create figure without showing it yet
        fig = plt.figure(figsize=(10, 6))
        plt.plot(result_df['k'], result_df['parametric_p_value'], label='Parametric P-Value',
                 marker='o', linestyle='-', color='blue')
        plt.plot(result_df['k'], result_df['non_parametric_p_value'], label='Non-Parametric P-Value',
                 marker='s', linestyle='--', color='orange')
        plt.xlabel('k Values')
        plt.ylabel('P-Values')
        plt.title(f'P-Values vs. k ({name})')
        plt.axhline(y=0.05, color='red', linestyle=':', label='Significance Threshold (p=0.05)')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save figure
        plt.savefig(f'./tuner_output_{name.lower()}/p_values_vs_k.png')
        figures.append(fig)

    # Create a summary plot comparing all kernels
    plt.figure(figsize=(12, 8))
    for name, result_df in results.items():
        plt.plot(result_df['k'], result_df['parametric_p_value'], marker='o', linestyle='-',
                 label=f'{name} (Parametric)')

    plt.xlabel('k Values')
    plt.ylabel('P-Values')
    plt.title('Parametric P-Values Comparison Across Kernels')
    plt.axhline(y=0.05, color='red', linestyle=':', label='Significance Threshold (p=0.05)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('./tuner_output_summary/kernel_comparison.png')

    # Display all figures
    for fig in figures:
        plt.figure(fig.number)
        plt.show()

    # Show the summary plot
    plt.show()

    return results
