from grakel import Graph, ShortestPath
from grakel.kernels import WeisfeilerLehman
from thesisv3.analysis.analysis import get_sub_distance_matrix, spectral_partition, dist_mat_to_graph, \
    kernighan_lin_partition
from thesisv3.building.building import construct_graph
import pandas as pd
from grakel.kernels import LovaszTheta
import networkx as nx
from tqdm import tqdm
from grakel.utils import graph_from_networkx


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
    edge_weight_tag = "inv_weight" if isinstance(kernel, ShortestPath) else "weight"
    if isinstance(kernel, ShortestPath):
        print(edge_weight_tag)
    # Convert each piece's segments into graphs
    # for piece, dist_mat in pieces_dist_mat.items():
    for piece in tqdm(pieces_dist_mat.keys(), desc="Processing within-piece similarities"):
        dist_mat = pieces_dist_mat[piece]
        if len(dist_mat) < minimum_segments:
            print(f"Skipping {piece}: insufficient segments ({len(dist_mat)})")
            continue

        partition1, partition2 = kernighan_lin_partition(pieces_graph_dict[piece], dist_mat, 42)

        # Compute WL kernel similarity for within-piece (same piece)
        grakel_graphs = [nx_to_grakel(G) for G in [partition1, partition2]]

        # grakel_graphs = graph_from_networkx(
        #     [partition1, partition2],
        #     node_labels_tag="label",
        #     edge_weight_tag=edge_weight_tag
        # )
        similarity_within = compare_graphs_kernel(grakel_graphs, kernel)[0, 1]
        between_piece_scores.append({
            'Piece_1': piece,
            'Piece_2': piece,
            'Between_Similarity': similarity_within
        })

    # Compare between pieces
    piece_names = list(pieces_dist_mat.keys())
    total_comparisons = len(piece_names) * (len(piece_names) - 1)
    with tqdm(total=total_comparisons, desc="Processing between-piece similarities") as pbar:
        for i in range(len(piece_names)):
            for j in range(len(piece_names)):
                piece_1, piece_2 = piece_names[i], piece_names[j]

                if piece_1 == piece_2:  # intra-graph similarity is already computed above, using partitions
                    continue

                full_graph1 = pieces_graph_dict[piece_1]
                full_graph2 = pieces_graph_dict[piece_2]

                # Compute WL kernel similarity between graph_1 of piece1 and graph_1 of piece2
                grakel_graphs = [nx_to_grakel(G) for G in [full_graph1, full_graph2]]
                # grakel_graphs = graph_from_networkx(
                #     [full_graph1, full_graph2],
                #     node_labels_tag="label",
                #     edge_weight_tag=edge_weight_tag
                # )
                similarity_between = compare_graphs_kernel(grakel_graphs, kernel)[0, 1]
                between_piece_scores.append({
                    'Piece_1': piece_1,
                    'Piece_2': piece_2,
                    'Between_Similarity': similarity_between
                })
                pbar.update(1)

    # Convert lists of dictionaries to DataFrames for better labeling and analysis
    # within_piece_df = pd.DataFrame(within_piece_scores)
    between_piece_df = pd.DataFrame(between_piece_scores)

    return between_piece_df


def compare_within_and_between_pieces2(pieces_dist_mat,
                                       pieces_graph_dict,
                                       kernel,
                                       minimum_segments=11):
    """
    Returns an N×N DataFrame (indexed and columned by piece name) where
    - diag[p,p] = WL similarity of that piece's two partitions
    - offdiag[p,q] = WL similarity between the full graphs of p and q
    """
    # Determine the appropriate edge weight tag for this kernel
    edge_weight_tag = "inv_weight" if isinstance(kernel, ShortestPath) else "weight"

    # 1) Filter out too‑small pieces
    piece_names = [
        p for p, dm in pieces_dist_mat.items()
        if len(dm) >= minimum_segments
    ]

    # 2) Compute the "within‑piece" similarity for each piece
    within_sim = {}
    for piece in piece_names:
        G = pieces_graph_dict[piece]
        part1, part2 = kernighan_lin_partition(G, 42)
        gk_parts = list(graph_from_networkx(
            [part1, part2],
            node_labels_tag="label",
            edge_weight_tag=edge_weight_tag
        ))
        # fresh kernel for each call to avoid re‑use state
        within_sim[piece] = compare_graphs_kernel(
            gk_parts,
            type(kernel)(**kernel.get_params())
        )[0, 1]

    # 3) Build a list of the full graphs, convert them all at once
    full_graphs = [pieces_graph_dict[p] for p in piece_names]
    gk_full = list(graph_from_networkx(
        full_graphs,
        node_labels_tag="label",
        edge_weight_tag=edge_weight_tag
    ))
    # compute the full N×N WL‐kernel matrix in one shot
    K_full = compare_graphs_kernel(
        gk_full,
        type(kernel)(**kernel.get_params())
    )

    # 4) Assemble into a square DataFrame and overwrite its diagonal
    sim_df = pd.DataFrame(K_full,
                          index=piece_names,
                          columns=piece_names)
    for p in piece_names:
        sim_df.loc[p, p] = within_sim[p]

    return sim_df