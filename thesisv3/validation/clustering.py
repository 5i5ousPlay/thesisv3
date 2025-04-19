import networkx as nx
import matplotlib.pyplot as plt
from karateclub.graph_embedding.graph2vec import Graph2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from thesisv3.utils.helpers import get_piece_type
import plotly.express as px
import pandas as pd


class GraphKMeans:
    def __init__(self, files: list[str], graphs: list[nx.Graph], k: int):
        """
        :param files (array): an n-length list of file names whose indices / position correspond to the passed graphs
        :param graphs (array): an n-length list of graphs whose indices / position correspond to the passed file names
        :param k (int): k-value to be used for clustering (I.E the number of artists in the corpus)
        """
        if len(files) != len(graphs):
            raise ValueError(
                "File and graph array shape are mismatched. File names and indices must correspond to graphs.")
        self.kmeans = KMeans(n_clusters=k)
        self.files = files
        self.graphs = graphs
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

        # # Improve layout
        # fig.update_layout(
        #     height=700,
        #     width=900,
        #     hoverlabel=dict(
        #         bgcolor="white",
        #         font_size=12,
        #         font_family="Arial"
        #     )
        # )

        # Show the plot
        fig.show()
