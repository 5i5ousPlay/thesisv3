from grakel import Graph, ShortestPath
from grakel.kernels import WeisfeilerLehman
from thesisv3.analysis.analysis import get_sub_distance_matrix, spectral_partition, dist_mat_to_graph, \
    kernighan_lin_partition
from thesisv3.building.building import construct_graph
import pandas as pd
from grakel.kernels import LovaszTheta
import networkx as nx
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from grakel.utils import graph_from_networkx
from typing import Type, Optional
from grakel.kernels import ShortestPath


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


def compare_within_and_between_pieces(
        pieces_dist_mat,
        pieces_graph_dict,
        kernel_cls: Type,
        kernel_kwargs: Optional[dict] = None,
        minimum_segments: int = 11
) -> pd.DataFrame:
    """
    kernel_cls    : the GraKeL kernel _class_ you want to use (not an instance)
    kernel_kwargs: dict of parameters to pass to kernel_cls()
    """
    kernel_kwargs = dict(kernel_kwargs or {})

    # 1) Filter out too-small pieces
    piece_names = [p for p, dm in pieces_dist_mat.items() if len(dm) >= minimum_segments]

    between_piece_scores = []
    within_sim = {}

    # 2) Within‑piece sims
    for piece in tqdm_notebook(piece_names, desc="Within-piece", leave=False):
        piece_display = piece[:20] + "..." if len(piece) > 20 else piece

        G = pieces_graph_dict[piece]
        if kernel_cls is ShortestPath:
            for u, v, data in G.edges(data=True):
                # pop inv_weight if present; otherwise leave existing weight or default to 1
                data['weight'] = data.pop('inv_weight', data.get('weight', 1))

        part1, part2 = kernighan_lin_partition(G, 42)
        gk_parts = [nx_to_grakel(p) for p in (part1, part2)]

        # instantiate a fresh kernel for each within‑piece call
        k_inst = kernel_cls(**kernel_kwargs)
        sim = k_inst.fit_transform(gk_parts)[0, 1]

        within_sim[piece] = sim
        between_piece_scores.append({
            "Piece_1": piece,
            "Piece_2": piece,
            "Between_Similarity": sim
        })

    # 3) Full N×N between‑piece sims in one shot
    full_graphs = [nx_to_grakel(pieces_graph_dict[p]) for p in piece_names]
    k_full = kernel_cls(**kernel_kwargs)
    K_full = k_full.fit_transform(full_graphs)

    # 4) Unpack the off‑diagonals
    for i, p1 in enumerate(piece_names):
        for j, p2 in enumerate(piece_names):
            if i == j:
                continue
            between_piece_scores.append({
                "Piece_1": p1,
                "Piece_2": p2,
                "Between_Similarity": K_full[i, j]
            })

    return pd.DataFrame(between_piece_scores)
