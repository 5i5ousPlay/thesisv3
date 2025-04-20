import os
import pickle

import grakel
import grakel as gk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from thesisv3.validation.comparison import compare_within_and_between_pieces
from thesisv3.building.building import construct_graph


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
                 output_dir='./Output/tuner_output',
                 batcher_dir='./Output/batcher_output'):
        """
        Initializes a Tuner instance. Args: graph_kernel (grakel.kernels.Kernel): Graph kernel used for calculating
        within and between graph scores seed (int): RNG seed for reproducibility min_k (int): minimum k value to test
        max_k (int): maximum k value to test k_step (int): interval step for k value testing output_dir (str):
        directory to save progress and results batcher_output_dir (str, optional): path to the GraphBatcher output
        directory to use instead of processing files again
        """
        self.distance_matrices = []
        self.segments = []
        self.graph_kernel = graph_kernel
        self.seed = seed
        self.min_k = min_k
        self.max_k = max_k
        self.k_step = k_step

        # Track processed files
        self.processed_files = []

        # Create dictionaries for mapping files to their data
        self.segment_dict = {}
        self.distmat_dict = {}
        self.graph_dict = {}
        self.new_graph_dict = False

        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.batcher_dir = batcher_dir

        # Define paths for each pickle file
        self.segments_path = os.path.join(self.batcher_dir, 'segments.pkl')
        self.distmat_path = os.path.join(self.batcher_dir, 'distance_matrices.pkl')
        self.processed_files_path = os.path.join(self.batcher_dir, 'processed_files.pkl')
        self.results_path = os.path.join(self.output_dir, 'tuner_results.pkl')

        # Dictionary to store results for each k value
        self.results = {}

        # Flag to track if we're using batcher data
        self.load_progress()

    def _kernel_based_similarity(self, k: int, label: str = None):
        # Load appropriate knn-graph dictionary
        def safe_load(path, default_value):
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    raise ValueError(f"Failed to load graphs from {path}: {e}")
            return default_value

        # Check if we need to initialize graphs
        if not self.graph_dict:
            # First time running - need to get graphs
            if label:
                # Case 1: Custom label provided - construct new graphs
                print(f"Constructing graphs with custom label: {label}")
                for composer, distmat in self.distmat_dict.items():
                    self.graph_dict[composer] = construct_graph(k, distmat, self.segment_dict[composer], label=label)
                self.new_graph_dict = True
                self.label_used = label  # Remember which label we used
            else:
                # Case 2: No label - try to load from pickle
                print(f"Loading pre-constructed graphs for k={k}")
                graphs_path = os.path.join(self.batcher_dir, f'graphs_k{k}.pkl')
                self.graph_dict = safe_load(graphs_path, {})
                if not self.graph_dict:
                    raise ValueError(f"No graphs found for k={k} and failed to load from {graphs_path}")
                self.new_graph_dict = True
                self.label_used = None
        elif label != getattr(self, 'label_used', None):
            # We have graphs but the label changed - reconstruct
            print(f"Label changed from {getattr(self, 'label_used', None)} to {label}. Reconstructing graphs.")
            self.graph_dict = {}
            for composer, distmat in self.distmat_dict.items():
                self.graph_dict[composer] = construct_graph(k, distmat, self.segment_dict[composer], label=label)
            self.label_used = label
        else:
            # We already have the right graphs, just reuse them
            print(f"Reusing existing graphs with label: {getattr(self, 'label_used', None)}")

        # Get a single DataFrame containing BOTH within‑ and between‑piece sims
        pair_df = compare_within_and_between_pieces(self.distmat_dict, self.graph_dict, self.graph_kernel, minimum_segments=11)

        # Separate the two cases
        within_mask = pair_df['Piece_1'] == pair_df['Piece_2']
        within_scores = pair_df.loc[within_mask, 'Between_Similarity'].tolist()
        between_scores = pair_df.loc[~within_mask, 'Between_Similarity'].tolist()

        return within_scores, between_scores

    def calculate_graph_statistics(self, label: str = None) -> pd.DataFrame:
        """
        Calculates the normality of the within and between graph scores as well as the
        corresponding p-value for each k-value within the specified range, with
        progress tracking and saving.

        Returns:
            graph_statistics (pd.DataFrame): Dataframe containing the parametric and non-parametric
            p-values for each k-value within the specified range.
        """
        if not self.distance_matrices or not self.segments:
            raise ValueError("Need distance matrices and segments to tune")

        cols = ['k',
                'normality_within', 'normality_between',
                'average_within', 'average_between',
                'parametric_p', 'non_parametric_p',
                'cohens_d', 'auc']
        graph_statistics = pd.DataFrame(columns=cols)

        # Check if we have cached results for any k values
        for k in range(self.min_k, self.max_k, self.k_step):
            if k in self.results:
                print(f"Using cached results for k = {k}")
                graph_statistics.loc[len(graph_statistics)] = self.results[k]
                continue

            idx = len(graph_statistics)
            print(f"Calculating graph statistics at k = {k}")
            within, between = self._kernel_based_similarity(k, label)

            # Normality tests
            _, p_norm_within = shapiro(within)
            _, p_norm_between = shapiro(between)

            # Average Scores
            avg_within = np.mean(within)
            avg_between = np.mean(between)

            # Parametric test
            # equal_var=False to use Welch’s t-tes, which doesnt assume equal variances for safety
            _, p_ttest = ttest_ind(within, between, equal_var=False)

            # Non-parametric test
            # alternative='two-sided' because we don’t want to assume direction in advance
            # (we're expecting within-piece similarity to be higher)
            u_stat, p_value_non_parametric = mannwhitneyu(within, between, alternative='two-sided')

            # Effect sizes

            # Cohen's d
            n1, n2 = len(within), len(between)
            pooled_sd = np.sqrt(((n1 - 1) * np.var(within, ddof=1) +
                                 (n2 - 1) * np.var(between, ddof=1)) / (n1 + n2 - 2))
            cohens_d = np.inf if pooled_sd==0 else (avg_within - avg_between) / pooled_sd

            # AUC (common‑language effect size)
            auc = u_stat / (n1 * n2)

            # Store the results
            result = [k, p_norm_within, p_norm_between,
                      avg_within, avg_between,
                      p_ttest, p_value_non_parametric,
                      cohens_d, auc]
            graph_statistics.loc[idx] = result

            # Cache the results
            self.results[k] = result

            # Save results after each k value
            self.save_progress()

        for col in ['parametric_p', 'non_parametric_p']:
            pvals = graph_statistics[col].values
            _, pvals_adj, _, _ = multipletests(pvals, method='fdr_bh')
            graph_statistics[f'{col}_adj'] = pvals_adj

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

        plt.plot(graph_statistics['k'], graph_statistics['parametric_p'], label='Parametric P-Value',
                 marker='o', linestyle='-', color='blue')
        plt.plot(graph_statistics['k'], graph_statistics['non_parametric_p'], label='Non-Parametric P-Value',
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


def compare_kernels(batcher_dir='./Output/batcher_output', output_dir='./Output/tuner_output', label=None, min_k=2, max_k=10, k_step=1):
    """
    Benchmark several graph‑kernel families on the **same** music dataset,
    plotting FDR‑corrected p‑values (and saving the raw DataFrames).

    Returns
    -------
    dict
        Maps kernel‑name → results DataFrame (columns include *parametric_p,
        parametric_p_adj, non_parametric_p, non_parametric_p_adj, cohens_d, auc*).
    """
    # ── 1. Kernels to evaluate ────────────────────────────────────────────────
    kernels = {
        # Weisfeiler-Lehman doesn't use edge weights, but we keep both normalized and raw versions
        'WeisfeilerLehman': gk.WeisfeilerLehman(n_iter=5, normalize=True),
        # 'WeisfeilerLehman (raw)': gk.WeisfeilerLehman(n_iter=5, normalize=False),

        # Shortest Path
        # 'ShortestPath': gk.ShortestPath(normalize=True, with_labels=True),
        # 'ShortestPath (Attr)': gk.ShortestPath(normalize=True),


        # "VertexHistogram": gk.VertexHistogram(normalize=True),
        # "PyramidMatch": gk.PyramidMatch(normalize=True),
        # "RandomWalkLabeled": gk.RandomWalkLabeled(lamda=0.1, method_type="fast", kernel_type="geometric")


        # Random Walk — edge weights are implicitly used via transition probabilities
        # 'RandomWalkLabeled (default)': gk.RandomWalkLabeled(lamda=0.1, method_type='fast', kernel_type='geometric'),

        # # GraphletSampling does not use edge weights — we keep it for completeness
        # 'GraphletSampling (norm)': gk.GraphletSampling(normalize=True),
        # 'GraphletSampling (raw)': gk.GraphletSampling(normalize=False),
    }

    results, figs = {}, []

    if label is not None:
        # Replace | with _ for directory naming
        label_dir = label.replace('|', '_')
        output_dir = f"{output_dir}_{label_dir}"

    # ── 2. Loop over kernels ─────────────────────────────────────────────────
    for name, kernel in kernels.items():
        print(f"\n{'=' * 60}\nTesting kernel: {name}\n{'=' * 60}\n")

        tuner = KNNGraphTuner(
            graph_kernel=kernel,
            batcher_dir=batcher_dir,
            min_k=min_k,
            max_k=max_k,
            k_step=k_step,
            output_dir=os.path.join(output_dir, name.lower().replace(" ", "_"))
        )

        df = tuner.calculate_graph_statistics(label=label)
        results[name] = df  # keep for later

        # ── Plot FDR‑corrected p‑values for this kernel ──────────────────────
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df['k'], df['parametric_p_adj'], 'o-', label='Parametric p (FDR)', lw=1.8)
        plt.plot(df['k'], df['non_parametric_p_adj'], 's--', label='Non-param p (FDR)', lw=1.8)
        plt.plot(df['k'], df['parametric_p'], 'o-', label='Parametric p', lw=1.8)
        plt.plot(df['k'], df['non_parametric_p'], 's--', label='Non-param p', lw=1.8)
        plt.axhline(0.05, ls=':', color='red', label='α = 0.05')
        plt.xlabel('k (neighbours)')
        plt.ylabel('FDR-corrected p-value')
        plt.title(f'FDR-corrected p-values vs k — {name}')
        plt.grid(ls='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, name.lower().replace(" ", "_"), 'p_values_vs_k.png'))
        figs.append(fig)

    # ── 3. Summary plot: raw vs. adjusted parametric p‑values ──────────────────
    plt.figure(figsize=(12, 7))

    for name, df in results.items():
        # Solid line: FDR-adjusted
        plt.plot(df['k'], df['parametric_p_adj'], marker='o', lw=1.8, label=f'{name} (adj)')
        # Dashed line: raw
        plt.plot(df['k'], df['parametric_p'], marker='x', lw=1.2, ls='--', label=f'{name} (raw)')

    plt.axhline(0.05, ls=':', color='red', label='α = 0.05')
    plt.xlabel('k (neighbours)')
    plt.ylabel('Parametric p‑value')
    plt.title('Kernel comparison: raw vs. FDR‑adjusted parametric p‑values')
    plt.grid(ls='--', alpha=.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    os.makedirs(os.path.join(output_dir, 'summary'), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary', 'kernel_comparison_raw_vs_adj.png'))

    # ── 4. Show individual figures (optional) ────────────────────────────────
    for fig in figs:
        fig.show()

    return results
