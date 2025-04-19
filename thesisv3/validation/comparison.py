from grakel import Graph
from grakel.kernels import WeisfeilerLehman
from thesisv3.analysis.analysis import get_sub_distance_matrix, spectral_partition, dist_mat_to_graph, \
    kernighan_lin_partition
from thesisv3.building.building import construct_graph
import pandas as pd
from grakel.kernels import LovaszTheta


def nx_to_grakel(G):
    edges = list(G.edges())
    labels = {node: idx for idx, node in enumerate(G.nodes())}
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


def compare_within_and_between_pieces(pieces_dist_mat, pieces_graph_dict, kernel, minimum_segments=10):
    """
    Compares within-piece and between-piece similarity scores, labeling them accordingly.

    Parameters:
    - pieces_dist_mat (dict): A dictionary where keys are piece names and values are distance matrices.
    - pieces_graph_dict (dict): A dictionary where keys are piece names and values are graphs.
    - k (int): Number of neighbors for k-NN graph.

    Returns:
    - between_piece_df (pd.DataFrame): DataFrame with pairs of piece names and their between-piece similarity scores.
    """
    between_piece_scores = []

    # Convert each piece's segments into graphs
    for piece, dist_mat in pieces_dist_mat.items():
        if len(dist_mat) < minimum_segments:
            print(f"Skipping {piece}: insufficient segments ({len(dist_mat)})")
            continue

        partition1, partition2 = kernighan_lin_partition(pieces_graph_dict[piece], dist_mat, 42)
        # Compute WL kernel similarity for within-piece (same piece)
        similarity_within = compare_graphs_kernel([partition1, partition2], kernel)[0, 1]
        between_piece_scores.append({
            'Piece_1': piece,
            'Piece_2': piece,
            'Between_Similarity': similarity_within
        })

    # Compare between pieces
    piece_names = list(pieces_dist_mat.keys())
    for i in range(len(piece_names)):
        for j in range(i+1, len(piece_names)):
            piece_1, piece_2 = piece_names[i], piece_names[j]

            if piece_1 == piece_2:  # intra-graph similarity is already computed above, using partitions
                continue

            full_graph1 = pieces_graph_dict[piece_1]
            full_graph2 = pieces_graph_dict[piece_2]

            # Compute WL kernel similarity between graph_1 of piece1 and graph_1 of piece2
            similarity_between = compare_graphs_kernel([full_graph1, full_graph2], kernel)[0, 1]
            between_piece_scores.append({
                'Piece_1': piece_1,
                'Piece_2': piece_2,
                'Between_Similarity': similarity_between
            })

    # Convert lists of dictionaries to DataFrames for better labeling and analysis
    # within_piece_df = pd.DataFrame(within_piece_scores)
    between_piece_df = pd.DataFrame(between_piece_scores)

    return between_piece_df
