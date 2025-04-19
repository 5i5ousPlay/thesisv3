from grakel import Graph
from grakel.kernels import WeisfeilerLehman
from thesisv3.analysis.analysis import get_sub_distance_matrix, spectral_partition, dist_mat_to_graph
import pandas as pd


def nx_to_grakel(G):
    edges = list(G.edges())
    labels = {node: idx for idx, node in enumerate(G.nodes())}
    return Graph(edges, node_labels=labels)


def compare_graphs(graph_list):
    grakel_graphs = [nx_to_grakel(g) for g in graph_list]

    # Initialize the Weisfeiler-Lehman kernel
    wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True)

    # Compute the WL kernel similarity matrix
    similarity_matrix = wl_kernel.fit_transform(grakel_graphs)

    return similarity_matrix


def compare_within_and_between_pieces(pieces_dist_mat, k, minimum_segments=30):
    """
    Compares within-piece and between-piece similarity scores, labeling them accordingly.

    Parameters:
    - pieces_dist_mat (dict): A dictionary where keys are piece names and values are distance matrices.
    - k (int): Number of neighbors for k-NN graph.

    Returns:
    - within_piece_df (pd.DataFrame): DataFrame with piece names and their within-piece similarity scores.
    - between_piece_df (pd.DataFrame): DataFrame with pairs of piece names and their between-piece similarity scores.
    """
    within_piece_scores = []
    between_piece_scores = []

    # Convert each piece's segments into graphs
    partitioned_piece_graphs = {}
    full_piece_graphs = {}
    for piece, dist_mat in pieces_dist_mat.items():
        if len(dist_mat) < minimum_segments:
            print(f"Skipping {piece}: insufficient segments ({len(dist_mat)})")
            continue
        group1, group2 = spectral_partition(dist_mat)

        D_group1 = get_sub_distance_matrix(dist_mat, group1)
        D_group2 = get_sub_distance_matrix(dist_mat, group2)

        # Generate k-NN graphs for each partition
        graph_1 = dist_mat_to_graph(k, D_group1)
        graph_2 = dist_mat_to_graph(k, D_group2)
        full_graph = dist_mat_to_graph(k, dist_mat)

        # Store the graphs for later comparisons
        partitioned_piece_graphs[piece] = (graph_1, graph_2)
        full_piece_graphs[piece] = full_graph

        # Compute WL kernel similarity for within-piece (same piece)
        similarity_within = compare_graphs([graph_1, graph_2])[0, 1]
        between_piece_scores.append({
            'Piece_1': piece,
            'Piece_2': piece,
            'Between_Similarity': similarity_within
        })

    # Compare between pieces
    piece_names = list(partitioned_piece_graphs.keys())
    for i in range(len(piece_names)):
        for j in range(len(piece_names)):
            piece_1, piece_2 = piece_names[i], piece_names[j]

            if piece_1 == piece_2:  # intra-graph similarity is already computed above, suing partitions
                continue

            full_graph1 = full_piece_graphs[piece_1]
            full_graph2 = full_piece_graphs[piece_2]

            # Compute WL kernel similarity between graph_1 of piece1 and graph_1 of piece2
            similarity_between = compare_graphs([full_graph1, full_graph2])[0, 1]
            between_piece_scores.append({
                'Piece_1': piece_1,
                'Piece_2': piece_2,
                'Between_Similarity': similarity_between
            })

    # Convert lists of dictionaries to DataFrames for better labeling and analysis
    # within_piece_df = pd.DataFrame(within_piece_scores)
    between_piece_df = pd.DataFrame(between_piece_scores)

    return between_piece_df
