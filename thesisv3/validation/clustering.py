import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
from karateclub.graph_embedding.graph2vec import Graph2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from thesisv3.utils.helpers import get_piece_type


class GraphKMeans:
    def __init__(self, files: list[str], graphs: list[nx.Graph], k: int, random_state: int = 42):
        """
        :param files (array): an n-length list of file names whose indices / position correspond to the passed graphs
        :param graphs (array): an n-length list of graphs whose indices / position correspond to the passed file names
        :param k (int): k-value to be used for clustering (I.E the number of artists in the corpus)
        :param random_state (int): Random seed for reproducibility
        """
        if len(files) != len(graphs):
            raise ValueError(
                "File and graph array shape are mismatched. File names and indices must correspond to graphs.")

        # Set random state for reproducible results
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=k, random_state=random_state)
        self.files = files
        self.graphs = graphs

        # Try to set random state for Graph2Vec
        np.random.seed(random_state)  # Global seed for numpy

        # Different Graph2Vec implementations might use different parameter names
        try:
            self.model = Graph2Vec(random_state=random_state)
        except TypeError:
            try:
                self.model = Graph2Vec(seed=random_state)
            except TypeError:
                # Fallback if no random state parameter is available
                self.model = Graph2Vec()

        self.embeddings = self._get_embeddings()
        self.clustered_graphs, self.labels = self.get_labels()

    def _get_embeddings(self):
        self.model.fit(self.graphs)
        embeddings = self.model.get_embedding()
        self.embeddings = embeddings
        return embeddings

    def get_labels(self):
        labels = self.kmeans.fit_predict(self.embeddings)
        clustered_graphs = pd.DataFrame(data={"file": self.files,
                                              "label": labels})
        return clustered_graphs, labels

    def find_optimal_k(self, k_range=range(2, 11)):
        """
        Find optimal number of clusters using the elbow method and silhouette analysis.

        :param k_range: Range of k values to test
        :return: Dictionary with metrics for each k value
        """
        from sklearn.metrics import silhouette_score
        import matplotlib.pyplot as plt

        results = {}
        X = self.embeddings

        # Metrics for each k value
        inertias = []
        silhouette_scores = []

        for k in k_range:
            # Create and fit KMeans with current k
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(X)

            # Calculate metrics
            inertia = kmeans.inertia_
            sil_score = silhouette_score(X, labels) if k > 1 else 0

            results[k] = {
                'inertia': inertia,
                'silhouette_score': sil_score
            }

            inertias.append(inertia)
            silhouette_scores.append(sil_score)

        # Plot elbow method
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(list(k_range), inertias, 'bo-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(list(k_range)[1:], silhouette_scores[1:], 'ro-')  # Skip k=1
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis for Optimal k')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return results

    def consensus_clustering(self, n_iterations=10, consensus_k=None):
        """
        Perform consensus clustering by running KMeans multiple times and
        finding the most consistent clustering.

        :param n_iterations: Number of clustering iterations to run
        :param consensus_k: K value for the final consensus clustering (defaults to self.kmeans.n_clusters)
        :return: Consensus cluster labels
        """
        import numpy as np
        from sklearn.cluster import KMeans

        X = self.embeddings
        n_samples = X.shape[0]
        k = self.kmeans.n_clusters if consensus_k is None else consensus_k

        # Connectivity matrix (co-occurrence matrix)
        connectivity_matrix = np.zeros((n_samples, n_samples))

        # Run KMeans multiple times with different initializations
        for i in range(n_iterations):
            # Create a new KMeans instance with a different random state
            iter_seed = self.random_state + i if self.random_state is not None else None
            kmeans = KMeans(n_clusters=k, random_state=iter_seed)
            labels = kmeans.fit_predict(X)

            # Update connectivity matrix
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    # If two samples are in the same cluster, increment their connection
                    if labels[i] == labels[j]:
                        connectivity_matrix[i, j] += 1
                        connectivity_matrix[j, i] += 1

        # Convert to similarity matrix (divide by number of iterations)
        similarity_matrix = connectivity_matrix / n_iterations

        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix

        # Perform final clustering on the consensus matrix
        from sklearn.cluster import AgglomerativeClustering
        final_clustering = AgglomerativeClustering(
            n_clusters=k,
            # affinity='precomputed',
            linkage='average'
        ).fit(distance_matrix)

        consensus_labels = final_clustering.labels_

        # Update class attributes with consensus results
        consensus_clustered_graphs = pd.DataFrame(data={"file": self.files,
                                                        "label": consensus_labels})

        # Store original labels
        self.original_labels = self.labels
        self.original_clustered_graphs = self.clustered_graphs

        # Update with consensus labels
        self.labels = consensus_labels
        self.clustered_graphs = consensus_clustered_graphs

        return consensus_labels

    def evaluate_clustering_quality(self):
        """
        Evaluate clustering quality using internal metrics (no ground truth needed).

        :return: Dictionary of evaluation metrics
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        # Get embeddings and labels
        X = self.embeddings
        labels = self.labels

        # Calculate metrics
        metrics = {
            'silhouette_score': silhouette_score(X, labels),
            'davies_bouldin_index': davies_bouldin_score(X, labels),
            'calinski_harabasz_index': calinski_harabasz_score(X, labels),
            'inertia': self.kmeans.inertia_  # Sum of squared distances to centroids
        }

        return metrics

    def visualize(self, display_file_name=False):
        pca_embed = PCA(n_components=2).fit_transform(self.embeddings)

        fig, ax = plt.subplots()
        ax.scatter(pca_embed[:, 0], pca_embed[:, 1], c=self.labels, cmap='viridis', s=50)
        if display_file_name:
            for i, txt in enumerate(self.files):
                ax.annotate(txt, (pca_embed[i, 0], pca_embed[i, 1]))
        plt.show()

    def visualize_interactive(self, display_file_name=False):
        """
        Creates an interactive visualization of the graph clusters
        with hover functionality showing composer, piece name, and piece type.

        Args:
            display_file_name (bool): Whether to display file names as text labels on the plot
        """
        # Get clustering results
        pca_embed = PCA(n_components=2).fit_transform(self.embeddings)

        # Extract composer, piece name, and piece type for each file
        composers = []
        piece_names = []
        piece_types = []

        for filename in self.files:
            # Parse filename to get composer and piece name
            parts = filename.split(' | ')
            composer = parts[0]
            piece_name = parts[1] if len(parts) > 1 else "Unknown"

            # Get piece type using the provided function
            piece_type = get_piece_type(filename)

            composers.append(composer)
            piece_names.append(piece_name)
            piece_types.append(piece_type)

        df = pd.DataFrame({
            'PCA1': pca_embed[:, 0],
            'PCA2': pca_embed[:, 1],
            'Cluster': self.labels,
            'Composer': composers,
            'Piece': piece_names,
            'Type': piece_types,
            'Filename': self.files
        })

        df['Cluster'] = df['Cluster'].astype(str)
        # Create the interactive scatter plot
        fig = px.scatter(
            df, x='PCA1', y='PCA2',
            color='Cluster',
            hover_data=['Composer', 'Piece', 'Type'],
            labels={'Cluster': 'Cluster Label'},
            title='Musical Piece Graph Clustering Visualization',
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        # Add file names as text labels if requested
        if display_file_name:
            fig.update_traces(
                text=df['Filename'],
                mode='markers+text',
                textposition='top center'
            )

        # Show the plot
        fig.show()
