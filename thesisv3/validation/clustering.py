import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from karateclub.graph_embedding.graph2vec import Graph2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


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
        self.embeddings = None

    def _get_embeddings(self):
        self.model.fit(self.graphs)
        embeddings = self.model.get_embedding()
        self.embeddings = embeddings
        return embeddings

    def get_labels(self):
        embeddings = self._get_embeddings()
        labels = self.kmeans.fit_predict(embeddings)
        clustered_graphs = pd.DataFrame(data={"file": self.files,
                                              "label": labels})
        return clustered_graphs, labels

    def visualize(self, display_file_name=False):
        clustered_graphs, labels = self.get_labels()
        pca_embed = PCA(n_components=2).fit_transform(self.embeddings)

        fig, ax = plt.subplots()
        ax.scatter(pca_embed[:, 0], pca_embed[:, 1], c=labels, cmap='viridis', s=50)
        if display_file_name:
            for i, txt in enumerate(self.files):
                ax.annotate(txt, (pca_embed[i, 0], pca_embed[i, 1]))
        plt.show()
