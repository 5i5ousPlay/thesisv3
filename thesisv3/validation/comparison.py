from grakel import Graph
from grakel.kernels import WeisfeilerLehman
from thesisv3.analysis.analysis import get_sub_distance_matrix, spectral_partition, dist_mat_to_graph, \
    kernighan_lin_partition
from thesisv3.building.building import construct_graph
import pandas as pd
from grakel.kernels import LovaszTheta
import networkx as nx


# def nx_to_grakel(G: nx.Graph) -> Graph:
#     """
#     Convert a (possibly weighted) NetworkX graph to a GraKeL Graph.
#     Keeps:
#       • edge attribute 'weight'  (falls back to 1.0)
#       • dummy integer node labels if none exist
#     """
#     # edge list with weights  →  (u, v, w)
#     edges = [(u, v, G[u][v].get("weight", 1.0)) for u, v in G.edges()]
#
#     # node labels: use existing 'label', else dummy ints
#     labels = nx.get_node_attributes(G, "label")
#     if not labels:
#         labels = {n: i for i, n in enumerate(G.nodes())}
#
#     return Graph(edges, node_labels=labels, graph_format="all")


def nx_to_grakel(G: nx.Graph) -> Graph:
    """
    Convert NetworkX graph to GraKeL Graph with:
    - Sequential 0-based node indexing
    - Dummy integer node labels if none are provided
    - Weighted edges preserved (default weight = 1.0)
    """

    # Map original node IDs to dense 0-based indices
    node_id_map = {node: i for i, node in enumerate(G.nodes())}

    # Relabel graph temporarily to ensure 0-based indexing
    G_reindexed = nx.relabel_nodes(G, node_id_map, copy=True)

    # Assign labels
    labels = nx.get_node_attributes(G, "label")
    if labels:
        node_labels = {node_id_map[n]: str(labels[n]) for n in G.nodes()}
    else:
        node_labels = {i: str(i) for i in G_reindexed.nodes()}

    # Build edge list with weights
    edges = []
    for u, v in G.edges():
        u_idx = node_id_map[u]
        v_idx = node_id_map[v]
        w = G[u][v].get("weight", 1.0)
        edges.append((u_idx, v_idx, w))

    return Graph(edges, node_labels=node_labels, graph_format="all")



def compare_graphs_kernel(grakel_graphs: list, graph_kernel):
    kernel = graph_kernel
    if isinstance(kernel, LovaszTheta):
        max_dim = max(len(g[0]) for g in grakel_graphs)
        similarity_matrix = LovaszTheta(normalize=True, max_dim=max_dim).fit_transform(grakel_graphs)
        return similarity_matrix
    similarity_matrix = kernel.fit_transform(grakel_graphs)
    return similarity_matrix


def compare_within_and_between_pieces(pieces_dist_mat, pieces_graph_dict, kernel, minimum_segments=11):
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
        grakel_graphs = [nx_to_grakel(G) for G in [partition1, partition2]]
        similarity_within = compare_graphs_kernel(grakel_graphs, kernel)[0, 1]
        between_piece_scores.append({
            'Piece_1': piece,
            'Piece_2': piece,
            'Between_Similarity': similarity_within
        })

    # Compare between pieces
    piece_names = list(pieces_dist_mat.keys())
    for i in range(len(piece_names)):
        for j in range(len(piece_names)):
            piece_1, piece_2 = piece_names[i], piece_names[j]

            if piece_1 == piece_2:  # intra-graph similarity is already computed above, using partitions
                continue

            full_graph1 = pieces_graph_dict[piece_1]
            full_graph2 = pieces_graph_dict[piece_2]

            # Compute WL kernel similarity between graph_1 of piece1 and graph_1 of piece2
            grakel_graphs = [nx_to_grakel(G) for G in [full_graph1, full_graph2]]
            similarity_between = compare_graphs_kernel(grakel_graphs, kernel)[0, 1]
            between_piece_scores.append({
                'Piece_1': piece_1,
                'Piece_2': piece_2,
                'Between_Similarity': similarity_between
            })

    # Convert lists of dictionaries to DataFrames for better labeling and analysis
    # within_piece_df = pd.DataFrame(within_piece_scores)
    between_piece_df = pd.DataFrame(between_piece_scores)

    return between_piece_df
