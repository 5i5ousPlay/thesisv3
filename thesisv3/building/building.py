from music21 import converter
from sklearn.manifold import MDS

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
# Precomputed cuts from your dataset:
length_cuts = [8, 14]
density_cuts = [2.0, 4.0]
diversity_cuts = [4, 5]
# expect_cuts = [0.3580, 0.4980, 0.5639, 0.6200]


expect_cuts = [0.36, 0.61, 0.67, 0.77]

def bin_length(n_notes):
    if n_notes <= length_cuts[0]:
        return "Short"
    if n_notes <= length_cuts[1]:
        return "Medium"
    return "Long"


def bin_density(density):
    if density <= density_cuts[0]:
        return "Sparse"
    if density <= density_cuts[1]:
        return "Moderate"
    return "Dense"


def bin_diversity(diversity):
    if diversity <= diversity_cuts[0]:
        return "Low"
    if diversity <= diversity_cuts[1]:
        return "Medium"
    return "High"


def bin_expectancy(e):
    labels = ["VeryLow", "Low", "Medium", "High", "VeryHigh"]
    for cut, lab in zip(expect_cuts, labels):
        if e <= cut:
            return lab
    return labels[-1]


def construct_graph(k: int, distance_matrix: np.ndarray, segments: list[pd.DataFrame],
                    force_connectivity: bool = True, label: str = 'expectancy|ir_mode') -> nx.Graph:
    num_segments = len(segments)
    if distance_matrix.shape != (num_segments, num_segments):
        raise ValueError("Shape of distance_matrix does not match the number of segments.")

    # k‑NN matrix whose entries already contain the DTW distance
    knn = kneighbors_graph(
        distance_matrix,
        n_neighbors=min(k, num_segments - 1),  # Ensure k is not larger than possible neighbors
        mode="distance",
        metric="precomputed",
        include_self=False  # Avoid self-loops initially
    )

    # Make it symmetric (kneighbors_graph is directed)
    knn = 0.5 * (knn + knn.T)

    # Build NX graph; keep the distance in edge attr "dist"
    G = nx.from_scipy_sparse_array(knn, edge_attribute="dist")

    # 4) Convert distance -> similarity weight (Gaussian)
    sigma = np.median(distance_matrix[distance_matrix > 0])
    for u, v, attr in G.edges(data=True):
        d = attr["dist"]
        d_clamped = max(d, 1e-8)
        # attr["weight"] = d
        gaussian = np.exp(-(d_clamped ** 2) / (2 * sigma ** 2))
        attr["weight"] = gaussian
        attr["inv_weight"] = 1 / gaussian

    # Ensure connectivity, preserving both attrs
    if not nx.is_connected(G) and force_connectivity:
        print(
            f"k={k}: k‑NN graph is disjoint with {nx.number_connected_components(G)} components. Ensuring connectivity…")
        comps = list(nx.connected_components(G))
        for i in range(len(comps) - 1):
            best = min(
                (
                    (n1, n2, distance_matrix[n1, n2])
                    for n1 in comps[i] for n2 in comps[i + 1]
                ),
                key=lambda x: x[2]  # choose closest pair
            )
            u, v, d = best
            G.add_edge(
                u, v,
                dist=d,
                weight=np.exp(-(d ** 2) / (2 * sigma ** 2)),
                inv_weight=1 / (np.exp(-(d ** 2) / (2 * sigma ** 2)))
            )
    label_types = label.split('|') if '|' in label else [label]

    feature_calculators = {
        'expectancy': lambda seg: bin_expectancy(float(seg['expectancy'].mean())),
        'ir_mode': lambda seg: seg['ir_symbol'].mode().iat[0] if not seg['ir_symbol'].mode().empty else "None",
        'segment_length_binned': lambda seg: bin_length(len(seg)),
        'rhythmic_density_binned': lambda seg: bin_density(
            len(seg) / max(seg['duration_beats'].sum(), 1e-6)  # Avoid division by zero
        ),
        'ir_pattern_diversity': lambda seg: bin_diversity(len(seg['ir_symbol'].unique())),
        'x': lambda seg: 'x',
        'index': lambda seg, idx: str(idx)
    }

    for idx, seg in enumerate(segments):
        if idx not in G:  # Node might have been isolated and not included if k=0 or disconnected
            G.add_node(idx)  # Ensure node exists

        # Calculate all possible features for this segment
        node_features = {}
        for feature_name, calculator in feature_calculators.items():
            try:
                if feature_name == 'index':
                    node_features[feature_name] = calculator(seg, idx)
                else:
                    node_features[feature_name] = calculator(seg)
            except Exception as e:
                print(f"Warning: Could not calculate feature '{feature_name}' for segment {idx}. Error: {e}")
                node_features[feature_name] = "Error"  # Assign error label

        # Construct the final label based on the requested types
        final_label_parts = []
        for label_type in label_types:
            label_type = label_type.strip()
            if label_type in node_features:
                final_label_parts.append(str(node_features[label_type]))  # Ensure string conversion
            else:
                print(f"Warning: Requested label type '{label_type}' not recognized. Skipping.")

        G.nodes[idx]['label'] = '|'.join(final_label_parts) if final_label_parts else "UnknownLabel"

        # Store raw expectancy as well
        try:
            G.nodes[idx]['expectancy'] = float(seg['expectancy'].mean())
        except:
            print("building expectancy porblem")
            G.nodes[idx]['expectancy'] = np.nan
    # print(f"Label: {final_label_parts}")
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
    G = construct_graph(k, distance_matrix, force_connect)

    pos = nx.spring_layout(G, seed=seed, iterations=iterations)
    nx.draw(G, node_size=50, pos=pos)

    if show_labels:
        labels = {i: str(i) for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title(graph_title + f" (K={k})")
    plt.show()


def distance_matrices_to_knn_graphs(k: int, distance_matrices: dict, segments: dict, seed: int, iterations: int,
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

    for ax, (composer, distance_matrix) in zip(axes_flat, distance_matrices.items()):
        G = construct_graph(k, distance_matrix, segments[composer], force_connectivity)

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
