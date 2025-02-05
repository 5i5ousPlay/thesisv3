# ===============================
# Graph Analysis
# ===============================

def spectral_partition(distance_matrix):
    """
    Performs spectral partitioning on a distance matrix.

    Args:
        distance_matrix (np.array): Pairwise distance matrix

    Returns:
        tuple: Two arrays containing indices for the partitioned groups
    """
    sigma = np.mean(distance_matrix[np.nonzero(distance_matrix)])
    similarity_matrix = np.exp(-distance_matrix ** 2 / (2. * sigma ** 2))
    np.fill_diagonal(similarity_matrix, 0)

    laplacian = csgraph.laplacian(similarity_matrix, normed=True)
    eigenvalues, eigenvectors = eigh(laplacian)
    fiedler_vector = eigenvectors[:, 1]

    partition = fiedler_vector > 0
    return np.where(partition)[0], np.where(~partition)[0]


def get_sub_distance_matrix(distance_matrix, group_indices):
    """
    Extracts a sub-matrix from a distance matrix based on group indices.

    Args:
        distance_matrix (np.array): Original distance matrix
        group_indices (np.array): Indices to extract

    Returns:
        np.array: Sub-matrix containing only the specified indices
    """
    return distance_matrix[np.ix_(group_indices, group_indices)]


def dist_mat_to_graph(k: int, distance_matrix):
    """
    Converts a distance matrix to a k-NN graph.

    Args:
        k (int): Number of nearest neighbors
        distance_matrix (np.array): Pairwise distance matrix

    Returns:
        networkx.Graph: K-nearest neighbors graph
    """
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
    return nx.from_scipy_sparse_array(knn_graph)


def nx_to_grakel(G):
    """
    Converts a NetworkX graph to a GraKeL graph format.

    Args:
        G (networkx.Graph): Input NetworkX graph

    Returns:
        grakel.Graph: Converted graph in GraKeL format
    """
    edges = list(G.edges())
    labels = {node: idx for idx, node in enumerate(G.nodes())}
    return Graph(edges, node_labels=labels)


def compare_graphs(graph_list):
    """
    Compares multiple graphs using the Weisfeiler-Lehman kernel.

    Args:
        graph_list (list): List of NetworkX graphs to compare

    Returns:
        np.array: Similarity matrix between graphs
    """
    grakel_graphs = [nx_to_grakel(g) for g in graph_list]
    wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True)
    return wl_kernel.fit_transform(grakel_graphs)


def compare_within_and_between_artists(artists_dist_mat, k, min_segments=60):
    """
    Analyzes similarities within and between artists using graph-based comparison.

    Args:
        artists_dist_mat (dict): Dictionary mapping artists to their distance matrices
        k (int): Number of neighbors for k-NN graph construction
        min_segments (int): Minimum number of segments required for analysis

    Returns:
        tuple: (within_artist_df, between_artist_df, stats_dict)
            - within_artist_df: DataFrame of within-artist similarities
            - between_artist_df: DataFrame of between-artist similarities
            - stats_dict: Dictionary of analysis statistics

    Raises:
        ValueError: If input parameters are invalid or no artists can be processed
    """
    if not isinstance(artists_dist_mat, dict) or not artists_dist_mat:
        raise ValueError("artists_dist_mat must be a non-empty dictionary")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    # Initialize tracking variables
    processed_artists = {}
    stats_dict = {
        'total_artists': len(artists_dist_mat),
        'processed_artists': 0,
        'skipped_artists': 0,
        'average_segments': 0
    }

    # Process each artist
    for artist, dist_mat in artists_dist_mat.items():
        try:
            if len(dist_mat) < min_segments:
                stats_dict['skipped_artists'] += 1
                print(f"Skipping {artist}: insufficient segments ({len(dist_mat)})")
                continue

            group1, group2 = spectral_partition(dist_mat)
            D_group1 = get_sub_distance_matrix(dist_mat, group1)
            D_group2 = get_sub_distance_matrix(dist_mat, group2)

            processed_artists[artist] = {
                'partitions': (
                    dist_mat_to_graph(k, D_group1),
                    dist_mat_to_graph(k, D_group2)
                ),
                'full_graph': dist_mat_to_graph(k, dist_mat),
                'num_segments': len(dist_mat)
            }

            stats_dict['processed_artists'] += 1
            stats_dict['average_segments'] += len(dist_mat)

        except Exception as e:
            print(f"Error processing artist {artist}: {str(e)}")
            stats_dict['skipped_artists'] += 1
            continue

    if not processed_artists:
        raise ValueError("No artists could be processed with the given parameters")

    # Calculate average segments
    if stats_dict['processed_artists'] > 0:
        stats_dict['average_segments'] /= stats_dict['processed_artists']

    # Compute similarities
    within_artist_scores = []
    between_artist_scores = []

    # Within-artist comparisons
    for artist, data in processed_artists.items():
        try:
            graph1, graph2 = data['partitions']
            similarity_within = compare_graphs([graph1, graph2])[0, 1]
            within_artist_scores.append({
                'Artist': artist,
                'Within_Similarity': similarity_within,
                'Num_Segments': data['num_segments']
            })
        except Exception as e:
            print(f"Error computing within-artist similarity for {artist}: {str(e)}")

    # Between-artist comparisons
    artist_names = list(processed_artists.keys())
    for i, artist_1 in enumerate(artist_names):
        for artist_2 in artist_names[i:]:
            try:
                full_graph1 = processed_artists[artist_1]['full_graph']
                full_graph2 = processed_artists[artist_2]['full_graph']

                similarity_between = compare_graphs([full_graph1, full_graph2])[0, 1]
                between_artist_scores.append({
                    'Artist_1': artist_1,
                    'Artist_2': artist_2,
                    'Between_Similarity': similarity_between,
                    'Segments_1': processed_artists[artist_1]['num_segments'],
                    'Segments_2': processed_artists[artist_2]['num_segments']
                })
            except Exception as e:
                print(f"Error computing between-artist similarity for {artist_1} and {artist_2}: {str(e)}")

    # Create DataFrames and add metadata
    timestamp = pd.Timestamp.now()
    within_artist_df = pd.DataFrame(within_artist_scores)
    between_artist_df = pd.DataFrame(between_artist_scores)

    within_artist_df['Analysis_Date'] = timestamp
    between_artist_df['Analysis_Date'] = timestamp

    # Update statistics
    stats_dict.update({
        'within_artist_mean': within_artist_df['Within_Similarity'].mean(),
        'between_artist_mean': between_artist_df['Between_Similarity'].mean(),
        'analysis_timestamp': timestamp
    })

    return within_artist_df, between_artist_df, stats_dict

