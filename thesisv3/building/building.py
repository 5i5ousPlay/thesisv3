from music21 import converter
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import os
from thesisv3.preprocessing.preprocessing import *
from thesisv3.utils import worker
from thesisv3.utils.helpers import *


# ===============================
# Distance Matrix Operations
# ===============================

def segments_to_distance_matrix(segments: list[pd.DataFrame], cores=None, debug=False):
    """
    Converts segments to a distance matrix using multiprocessing.

    Parameters:
    segments (list[pd.DataFrame]): A list of segmented DataFrames.
    cores (int): The number of CPU cores to use for multiprocessing (default is None).

    Returns:
    np.ndarray: A distance matrix representing distances between segments.
    """
    if cores is not None and cores > cpu_count():
        raise ValueError(f"You don't have enough cores! Please specify a value within your system's number of "
                         f"cores. Core Count: {cpu_count()}")

    seg_np = [segment.to_numpy() for segment in segments]

    num_segments = len(seg_np)
    distance_matrix = np.zeros((num_segments, num_segments))

    args_list = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            args_list.append((i, j, segments[i], segments[j]))

    with Manager() as manager:
        message_list = manager.list()

        def log_message(message):
            message_list.append(message)

        with Pool(cores) as pool:
            results = pool.map(worker.calculate_distance, args_list)

        for i, j, distance, message in results:
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Reflect along the diagonal
            log_message(message)

        if debug:
            for message in message_list:
                print(message)

    return distance_matrix


def segments_to_distance_matrices(segments: dict, pickle_dir=None, pickle_file=None):
    """
    Creates distance matrices for multiple composers' segments.

    Args:
        segments (dict): Dictionary mapping composers to their segments
        pickle_dir (str, optional): Directory to save pickle file
        pickle_file (str, optional): Custom filename for pickle file

    Returns:
        dict: Mapping of composers to their distance matrices
    """
    dist_mats = {}
    for composer, segments in segments.items():
        print(f'Composer: {composer} | Segments: {len(segments)}')
        dist_mats[composer] = segments_to_distance_matrix(segments)

    if pickle_dir:
        output_filename = pickle_file if pickle_file else 'composer_segments.pickle'
        save_to_pickle(dist_mats, os.path.join(pickle_dir, output_filename))

    return dist_mats


# ===============================
# Graph Construction & Visualization
# ===============================

def construct_graph(k: int, distance_matrix: np.ndarray) -> nx.Graph:
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
    G = nx.from_scipy_sparse_array(knn_graph)

    # Optional: Add labels from segments if needed
    # for i in range(len(segments)):
    #     G.nodes[i]['label'] = np.round(segments[i]['expectancy'].mean(), decimals=2)
    #     G.nodes[i]['label'] = i

    if not nx.is_connected(G):
        print("The KNN graph is disjoint. Ensuring connectivity...")
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
            G.add_edge(*closest_pair)

    return G

def distance_matrix_to_knn_graph(k: int, distance_matrix: np.array, graph_title: str,
                                 seed: int, iterations: int, force_connect=False, show_labels=False):
    """
    Creates and visualizes a k-nearest neighbors graph from a distance matrix.

    Args:
        k (int): Number of nearest neighbors
        distance_matrix (np.array): Pairwise distance matrix
        graph_title (str): Title for the graph
        seed (int): Random seed for layout
        iterations (int): Number of layout iterations
        force_connect (bool): Whether to force graph connectivity
        show_labels (bool): Whether to show node labels

    Returns:
        None (displays plot)
    """
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
    G = nx.from_scipy_sparse_array(knn_graph)

    if not nx.is_connected(G) and force_connect:
        print("Connecting disjoint graph components...")
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

    pos = nx.spring_layout(G, seed=seed, iterations=iterations)
    nx.draw(G, node_size=50, pos=pos)

    if show_labels:
        labels = {i: str(i) for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title(graph_title + f" (K={k})")
    plt.show()


def distance_matrices_to_knn_graphs(k: int, distance_matrices: dict, seed: int, iterations: int,
                                    save_figures: bool = False, output_dir: str = "./Output/figures",
                                    layout_type: str = "spring", force_connectivity: bool = False):
    """
    Creates KNN graphs from distance matrices and plots them in a grid layout.

    Parameters:
    k (int): Number of nearest neighbors for the KNN graph.
    distance_matrices (dict): Dictionary where keys are composer names and values are distance matrices.
    seed (int): Random seed for layout algorithms that use randomization.
    iterations (int): Number of iterations for iterative layout algorithms.
    save_figures (bool, optional): Whether to save individual figures. Defaults to False.
    output_dir (str, optional): Directory to save figures to. Defaults to "./Output/figures".
    layout_type (str, optional): Layout algorithm to use: "spring", "kamada", "spectral". Defaults to "spring".
    force_connectivity (bool, optional): Whether to force the graph to be connected. Defaults to False.
    """
    if save_figures and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_graphs = len(distance_matrices)
    rows_needed = (num_graphs + 1) // 2
    fig, axes = plt.subplots(rows_needed, 2, figsize=(12, rows_needed * 5))

    if rows_needed == 1 and num_graphs == 1:
        axes = np.array([[axes]])
    elif num_graphs > 1:
        axes = np.array(axes)
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

    axes_flat = axes.flatten()

    for i, (composer, distance_matrix) in enumerate(distance_matrices.items()):
        if i >= len(axes_flat):
            break

        ax = axes_flat[i]
        knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
        G = nx.from_scipy_sparse_array(knn_graph)

        # Check if we need to force connectivity
        if force_connectivity and not nx.is_connected(G):
            print(f"The KNN graph for {composer} is disjoint. Ensuring connectivity...")

            # Calculate the connected components
            components = list(nx.connected_components(G))

            # Connect the components
            for j in range(len(components) - 1):
                min_dist = np.inf
                closest_pair = None
                for node1 in components[j]:
                    for node2 in components[j + 1]:
                        dist = distance_matrix[node1, node2]
                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = (node1, node2)

                # Add an edge between the closest pair of nodes from different components
                G.add_edge(closest_pair[0], closest_pair[1])

        # Apply the selected layout algorithm
        if layout_type == "spring":
            pos = nx.spring_layout(G, seed=seed, iterations=iterations, scale=2.0, center=(0, 0))
        elif layout_type == "kamada":
            pos = nx.kamada_kawai_layout(G)
        elif layout_type == "spectral":
            pos = nx.spectral_layout(G)
        else:
            # Default to spring layout if invalid option
            pos = nx.spring_layout(G, seed=seed, iterations=iterations, scale=2.0, center=(0, 0))

        nx.draw(G, node_size=50, pos=pos, ax=ax)
        ax.set_title(f"{composer} (K={k})")
        ax.axis('off')

        if save_figures:
            fig_individual, ax_individual = plt.subplots(figsize=(6, 5))
            nx.draw(G, node_size=50, pos=pos, ax=ax_individual)
            ax_individual.set_title(f"{composer} (K={k})")
            ax_individual.axis('off')

            safe_filename = composer.replace('|', '-').replace(':', '-').replace('\\', '-').replace('/', '-').replace(
                '*', '-').replace('?', '-').replace('"', '-').replace('<', '-').replace('>', '-')
            output_path = os.path.join(output_dir, f"{safe_filename}_KNN_graph.png")
            fig_individual.savefig(output_path)
            plt.close(fig_individual)

    for ax in axes_flat[num_graphs:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def distance_matrix_to_knn_graph_scaled(k: int, distance_matrix: np.array, graph_title: str,
                                        seed: int):
    """
    Creates a KNN graph with node positions scaled according to the distance matrix.

    Args:
        k (int): Number of nearest neighbors
        distance_matrix (np.array): Pairwise distance matrix
        graph_title (str): Title for the graph
        seed (int): Random seed for reproducibility

    Returns:
        None (displays plot)
    """

    def adjust_overlapping_nodes(pos, threshold=0.01, adjustment=0.05):
        nodes = list(pos.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, node_j = nodes[i], nodes[j]
                xi, yi = pos[node_i]
                xj, yj = pos[node_j]
                distance = np.hypot(xi - xj, yi - yj)
                if distance < threshold:
                    pos[node_j] = (xj + adjustment, yj + adjustment)
        return pos

    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
    G = nx.from_scipy_sparse_array(knn_graph)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed)
    positions = mds.fit_transform(distance_matrix)
    pos = {i: positions[i] for i in range(len(positions))}
    pos = adjust_overlapping_nodes(pos, threshold=1, adjustment=0.5)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#4481FB")
    nx.draw_networkx_edges(G, pos)

    labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='white')

    plt.title(graph_title + f" (K={k})")
    plt.axis('equal')
    plt.axis('off')
    plt.show()


# ===============================
# Music Processing Functions
# ===============================

def mass_produce_segments(filepath, pickle_dir=None, pickle_file=None):
    """
    Processes multiple music files to generate segments for analysis.

    Args:
        filepath (str): Directory containing music files
        pickle_dir (str, optional): Directory to save pickle file
        pickle_file (str, optional): Custom filename for pickle file

    Returns:
        dict: Dictionary mapping composers to their segments

    Notes:
        This function assumes the existence of helper functions:
        - parse_score_elements
        - assign_ir_symbols
        - ir_symbols_to_matrix
        - assign_ir_pattern_indices
        - segmentgestalt
        - preprocess_segments
    """
    directories = os.listdir(filepath)
    composer_segments = dict.fromkeys(directories, None)
    piece_count = 0

    for piece in os.listdir(filepath):
        piece_path = os.path.join(filepath, piece)
        try:
            # Parse and process the music score
            parsed_score = converter.parse(piece_path)
            nmat, narr, sarr = parse_score_elements(parsed_score)
            ir_symbols = assign_ir_symbols(narr)
            #TODO: Expectancy score here
            ir_nmat = ir_symbols_to_matrix(ir_symbols, nmat)
            ir_nmat = assign_ir_pattern_indices(ir_nmat)

            # Generate and preprocess segments
            segments = segmentgestalt(ir_nmat)
            prepped_segments = preprocess_segments(segments)

            piece_count += 1
            print(f'Composer: {piece} | Piece Count: {piece_count} \n Processed Segments: {len(prepped_segments)}')

            composer_segments[piece] = prepped_segments

        except Exception as e:
            print(f"Error processing piece {piece}: {str(e)}")
            continue

    # Save results if directory is specified
    if pickle_dir:
        output_filename = pickle_file if pickle_file else 'composer_segments.pickle'
        save_to_pickle(composer_segments, os.path.join(pickle_dir, output_filename))

    return composer_segments
