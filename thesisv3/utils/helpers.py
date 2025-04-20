import pickle
import os
import networkx as nx
from grakel import Graph
from grakel.kernels import LovaszTheta
from thesisv3.preprocessing.preprocessing import segments_to_distance_matrix


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


def get_piece_type(piece_name):
    parts = piece_name.split('|')
    if len(parts) > 1:
        piece_part = parts[1].strip()

        # Special handling for Chopin's Études
        if 'Étude' in piece_part:
            if 'Op. 10' in piece_part:
                return "Chopin's Études Op. 10"
            elif 'Op. 25' in piece_part:
                return "Chopin's Études Op. 25"
            else:
                return 'Études (General)'
        elif 'Waltz' in piece_part:
            return "Chopin's Waltzes"
        elif 'Sonata' in piece_part:
            return "Ysaÿe's Violin Sonatas"
        elif 'Suite' in piece_part:
            return "Bach's Cello Suites"
        elif 'Ballade' in piece_part:
            return "Chopin's Ballades"
        else:
            # Default to first word
            return piece_part.split(' ')[0]
    return 'Unknown'


def concatenate_segments(segment_dict, graph_dict=None):
    """
    Process a dictionary of music segments and create a distance matrix with metadata,
    including node labels from the graph.

    Parameters:
    -----------
    segment_dict : dict
        Dictionary where keys are composer/piece strings in format "composer | piece name"
        and values are lists of dataframes representing segments

    graph_dict : dict, optional
        Dictionary where keys match segment_dict keys and values are the corresponding
        networkx graphs with node labels

    Returns:
    --------
    tuple
        (all_segments, segment_metadata, distance_matrix)
        - all_segments: List of all segment dataframes
        - segment_metadata: List of dictionaries with metadata for each segment
        - distance_matrix: NumPy array of pairwise distances between segments
    """
    # Create a consolidated list of all segments
    all_segments = []
    segment_metadata = []  # To track which composer and piece each segment belongs to
    segment_indices = {}  # Keep track of global indices for each piece

    for composer_piece, df_list in segment_dict.items():
        # Extract composer and determine piece type
        composer = composer_piece.split('|')[0].strip() if '|' in composer_piece else composer_piece
        piece_type = get_piece_type(composer_piece)

        # If we have a graph for this piece, get the node labels
        graph = graph_dict.get(composer_piece) if graph_dict else None

        # Start tracking indices for this piece
        segment_indices[composer_piece] = []

        # for every segment (node) in a piece (one graph)
        for segment_idx, df in enumerate(df_list):
            # Store the global index for this segment

            # Prepare metadata with optional node label
            metadata = {
                'composer': composer,
                'piece_name': composer_piece,
                'piece_type': piece_type,
                'segment_idx': segment_idx,
            }

            # Add node label if graph is available
            if graph and segment_idx in graph.nodes:  # Use piece_idx instead
                metadata['node_label'] = graph.nodes[segment_idx].get('label',
                                                                    f"Node {segment_idx}")  # Use piece_idx here too
                # Add other attributes using piece_idx for lookup
                for attr_key, attr_value in graph.nodes[segment_idx].items():
                    if attr_key != 'label':
                        metadata[f'node_{attr_key}'] = attr_value

            all_segments.append(df)
            segment_metadata.append(metadata)

    # Generate distance matrix
    distance_matrix = segments_to_distance_matrix(all_segments)

    return all_segments, segment_metadata, distance_matrix
