import os
import pickle
import sys
from contextlib import contextmanager
from typing import Type, Optional

import grakel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from grakel.kernels import (
    WeisfeilerLehman,
    ShortestPath,
    VertexHistogram,
    PyramidMatch,
    # GraphletSampling,
)
from scipy.stats import shapiro
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm.notebook import tqdm as tqdm_notebook

from thesisv3.building.building import construct_graph
from thesisv3.validation.comparison import compare_within_and_between_pieces


# Stream handler for redirecting prints to the progress bar
class TqdmStreamHandler:
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            self.buffer = lines.pop()
            for line in lines:
                if line.strip():  # Only update with non-empty lines
                    self.progress_bar.set_description(line[:40] + "..." if len(line) > 40 else line)

    def flush(self):
        pass


# Context manager to suppress inner tqdm
@contextmanager
def suppress_inner_tqdm():
    """Temporarily replace tqdm with a no-op version"""
    import tqdm as tqdm_module
    original_tqdm = tqdm_module.tqdm

    # Create a dummy tqdm that just returns the iterable
    def dummy_tqdm(iterable, **kwargs):
        return iterable

    # Add required methods
    dummy_tqdm.update = lambda *args, **kwargs: None
    dummy_tqdm.close = lambda: None
    dummy_tqdm.set_description = lambda *args, **kwargs: None

    try:
        # Replace tqdm in various places it might be imported
        tqdm_module.tqdm = dummy_tqdm
        if 'tqdm' in sys.modules:
            sys.modules['tqdm'].tqdm = dummy_tqdm
        yield
    finally:
        # Restore original tqdm
        tqdm_module.tqdm = original_tqdm
        if 'tqdm' in sys.modules:
            sys.modules['tqdm'].tqdm = original_tqdm


class KNNGraphTuner:
    """
    Tuner class for calculating the p-values for within and between graph scores
    for each corresponding k values within a specified range.
    """

    def __init__(self,
                 kernel_cls: Type[grakel.kernels.Kernel],
                 kernel_kwargs: Optional[dict] = None,
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
        self.kernel_cls = kernel_cls
        self.kernel_kwargs = dict(kernel_kwargs or {})

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
        self.distance_matrices = []
        self.segments = []

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
        # Check if we need to construct new graphs based on label or k value
        k_specific_graph_path = os.path.join(self.batcher_dir, f'graphs_k{k}.pkl')

        if label:
            # Case 1: Custom label provided - always construct new graphs
            print(f"Constructing graphs with custom label: {label}")
            for composer, distmat in self.distmat_dict.items():
                self.graph_dict[composer] = construct_graph(k, distmat, self.segment_dict[composer], label=label)
            self.using_custom_label = True  # Track that we're using a custom label
        else:
            # Case 2: No custom label - try to load from pickle if available
            self.using_custom_label = False
            if os.path.exists(k_specific_graph_path):
                print(f"Loading pre-constructed graphs for k={k}")
                self.graph_dict = safe_load(k_specific_graph_path, {})
                if not self.graph_dict:
                    print(f"Error loading graphs for k={k}")
                    return None, None
            else:
                # No pickle found, construct default graphs
                print(f"Constructing default graphs for k={k}")
                for composer, distmat in self.distmat_dict.items():
                    self.graph_dict[composer] = construct_graph(k, distmat, self.segment_dict[composer])

        # Get a single DataFrame containing BOTH within‑ and between‑piece sims
        pair_df = compare_within_and_between_pieces(
            self.distmat_dict,
            self.graph_dict,
            self.kernel_cls,
            self.kernel_kwargs,
            minimum_segments=11
        )

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

        # Create a progress bar for k values
        k_values = list(range(self.min_k, self.max_k, self.k_step))
        k_progress = tqdm_notebook(k_values, desc="Processing k values")

        # Check if we have cached results for any k values
        for k in k_progress:
            k_progress.set_description(f"Processing k = {k}")

            if k in self.results:
                k_progress.set_description(f"Using cached results for k = {k}")
                graph_statistics.loc[len(graph_statistics)] = self.results[k]
                continue

            idx = len(graph_statistics)
            # print(f"Calculating graph statistics at k = {k}")
            with suppress_inner_tqdm():
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
        plt.title(f'P-Values vs. k ({self.kernel_cls.__name__})')
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
        'WeisfeilerLehman': (
            WeisfeilerLehman,
            {'n_iter': 4, 'normalize': True}
        ),
        'WeisfeilerLehman (raw)': (
            WeisfeilerLehman,
            {'n_iter': 4, 'normalize': False}
        ),
        'ShortestPath': (
            ShortestPath,
            {'normalize': True, 'with_labels': True}
            # note: 'attribute': 'inv_weight' will be injected automatically in the tuner
        ),
        'VertexHistogram': (
            VertexHistogram,
            {'normalize': True}
        ),
        'PyramidMatch': (
            PyramidMatch,
            {'normalize': True}
        ),
        # 'RandomWalkLabeled': (
        #     RandomWalkLabeled,
        #     {'lamda': 0.1, 'method_type': 'fast', 'kernel_type': 'geometric'}
        # ),
        # if you later want GraphletSampling:
        # 'GraphletSampling (norm)': (
        #     GraphletSampling,
        #     {'normalize': True}
        # ),
        # 'GraphletSampling (raw)': (
        #     GraphletSampling,
        #     {'normalize': False}
        # ),
    }

    results, figs = {}, []

    if label is not None:
        # Replace | with _ for directory naming
        label_dir = label.replace('|', '_')
        output_dir = f"{output_dir}_{label_dir}"

    # ── 2. Loop over kernels ─────────────────────────────────────────────────
    main_progress = tqdm_notebook(kernels.items(), total=len(kernels))
    results = {}
    for name, (kernel_cls, kernel_kwargs) in main_progress:
        main_progress.set_description(f"Processing: {name}")
        # print(f"\n{'=' * 60}\nTesting kernel: {name}\n{'=' * 60}\n")

        original_stdout = sys.stdout
        sys.stdout = TqdmStreamHandler(main_progress)

        try:
            # Your code runs here, prints will update the progress bar
            tuner = KNNGraphTuner(
                kernel_cls=kernel_cls,
                kernel_kwargs=kernel_kwargs,
                batcher_dir=batcher_dir,
                min_k=min_k,
                max_k=max_k,
                k_step=k_step,
                output_dir=os.path.join(output_dir, name.lower().replace(" ", "_"))
            )

            df = tuner.calculate_graph_statistics(label=label)
            results[name] = df
        finally:
            # Restore stdout
            sys.stdout = original_stdout

        # ── Plot FDR‑corrected p‑values for this kernel ──────────────────────
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df['k'], df['parametric_p_adj'], 'o-', label='Parametric p (FDR)', lw=1.8)      # make the _adj have similar color
        plt.plot(df['k'], df['non_parametric_p_adj'], 's--', label='Non-param p (FDR)', lw=1.8)
        plt.plot(df['k'], df['parametric_p'], 'o-', label='Parametric p', lw=1.8)   # make the not adj have similar color
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

        # # Create a more readable multi-panel figure
        # fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
        # axes = axes.flatten()
        #
        # # Define a consistent color palette for all kernels
        #
        # cmap = plt.get_cmap('tab10')
        # colors = cmap(np.linspace(0, 1, len(kernels)))
        # kernel_colors = dict(zip(kernels.keys(), colors))
        #
        # # Plot 1: Parametric p-values (adjusted)
        # for i, (name, df) in enumerate(results.items()):
        #     axes[0].plot(df['k'], df['parametric_p_adj'], 'o-',
        #                  color=kernel_colors[name], label=name, lw=2)
        # axes[0].set_title('FDR-adjusted parametric p-values')
        # axes[0].axhline(0.05, ls=':', color='red')
        # axes[0].set_ylabel('p-value')
        #
        # # Plot 2: Non-parametric p-values (adjusted)
        # for name, df in results.items():
        #     axes[1].plot(df['k'], df['non_parametric_p_adj'], 's-',
        #                  color=kernel_colors[name], lw=2)
        # axes[1].set_title('FDR-adjusted non-parametric p-values')
        # axes[1].axhline(0.05, ls=':', color='red')
        #
        # # Plot 3: Effect sizes
        # for name, df in results.items():
        #     axes[2].plot(df['k'], df['cohens_d'], 'D-',
        #                  color=kernel_colors[name], lw=2)
        # axes[2].set_title("Cohen's d effect sizes")
        # axes[2].set_ylabel("Effect size")
        # axes[2].set_xlabel('k (neighbours)')
        #
        # # Plot 4: AUC values
        # for name, df in results.items():
        #     axes[3].plot(df['k'], df['auc'], '^-',
        #                  color=kernel_colors[name], lw=2)
        # axes[3].set_title('AUC values')
        # axes[3].set_xlabel('k (neighbours)')
        #
        # # Single legend for the entire figure
        # fig.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
        # plt.tight_layout()

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

    # def _kernel_based_similarity(self, k: int, label: str = None):
    #     # Load appropriate knn-graph dictionary
    #     print(f"Calculating similarities for k={k} with label='{label}'...")
    #     current_graphs = {}
    #     # Always construct graphs needed for this specific call
    #     print(f"Constructing graphs with k={k}, label='{label}'...")
    #     try:
    #         # Assuming self.distmat_dict and self.segment_dict are populated correctly
    #         for piece_name, distmat in self.distmat_dict.items():
    #             # Ensure segments exist for this piece
    #             if piece_name not in self.segment_dict:
    #                 print(f"Warning: Segments not found for piece '{piece_name}'. Skipping graph construction.")
    #                 continue
    #             # Check if distmat matches segment count (important!)
    #             num_segments_expected = len(self.segment_dict[piece_name])
    #             if distmat.shape[0] != num_segments_expected or distmat.shape[1] != num_segments_expected:
    #                 print(f"Warning: Distance matrix shape {distmat.shape} mismatch for '{piece_name}' "
    #                       f"with {num_segments_expected} segments. Skipping.")
    #                 continue
    #
    #             # Ensure enough segments for k-NN
    #             if num_segments_expected <= k:
    #                 print(f"Warning: Piece '{piece_name}' has {num_segments_expected} segments, "
    #                       f"which is <= k={k}. Skipping k-NN graph construction.")
    #                 continue
    #
    #             # Pass the global_boundaries if needed by construct_graph/bin functions
    #             current_graphs[piece_name] = construct_graph(
    #                 k,
    #                 distmat,
    #                 self.segment_dict[piece_name],
    #                 label=label,
    #                 # global_boundaries=self.global_boundaries # Pass if needed
    #             )
    #     except Exception as e:
    #         print(f"Error during graph construction for k={k}, label='{label}': {e}")
    #         raise  # Re-raise the exception to halt if construction fails critically
    #
    #     if not current_graphs:
    #         print(f"Warning: No graphs were constructed for k={k}, label='{label}'. Returning empty lists.")
    #         return [], []
    #
    #     # Get a single DataFrame containing BOTH within‑ and between‑piece sims
    #     pair_df = compare_within_and_between_pieces(
    #         # Pass only distmats for pieces where graphs were successfully created
    #         {name: self.distmat_dict[name] for name in current_graphs.keys()},
    #         current_graphs,  # Pass the graphs constructed in this call
    #         self.kernel_cls,
    #         self.kernel_kwargs,
    #         minimum_segments=11  # This filtering happens again inside, maybe filter earlier?
    #     )
    #
    #     if pair_df.empty:
    #         print(f"Warning: Comparison DataFrame is empty for k={k}, label='{label}'.")
    #         return [], []
    #
    #     # Separate the two cases
    #     within_mask = pair_df['Piece_1'] == pair_df['Piece_2']
    #     within_scores = pair_df.loc[within_mask, 'Between_Similarity'].tolist()
    #     between_scores = pair_df.loc[~within_mask, 'Between_Similarity'].tolist()
    #
    #     return within_scores, between_scores
