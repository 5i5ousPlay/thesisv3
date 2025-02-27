import pickle
import os
import traceback

import music21
from music21 import converter, environment

from thesisv3.analysis.visualization import *
from thesisv3.preprocessing.preprocessing import *
from thesisv3.building.building import distance_matrix_to_knn_graph_scaled
from thesisv3.utils.file_manager import MusicFileManager
from thesisv3.utils.helpers import save_to_pickle

# Configure MuseScore paths
env = environment.Environment()
env['musicxmlPath'] = 'C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe'
env['musescoreDirectPNGPath'] = 'C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe'

us = music21.environment.UserSettings()
us['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'

__all__ = ['MusicFileManager', 'MusicSegmentAnalyzer', 'MusicVisualizer']


class MusicSegmentAnalyzer:
    """Handles the analysis of musical segments"""

    def __init__(self, score_path=None):
        self.score_path = score_path
        self.parsed_score = None
        self.segments = None
        self.prepped_segments = None
        self.distance_matrix = None
        self.nmat = None  # create field for raw nmat
        self.narr = None
        self.sarr = None
        self.ir_symbols = None

    def load_score(self, score_path=None):
        """Load and parse a music score"""
        if score_path:
            self.score_path = score_path
        if not self.score_path:
            raise ValueError("No score path provided")

        # Check if the path is a directory or a file.
        if os.path.isdir(self.score_path):
            self.parsed_score = 'dir'
        elif os.path.isfile(self.score_path):
            self.parsed_score = converter.parse(self.score_path)
        else:
            raise ValueError("The provided path is neither a file nor a directory")

        return self

    def analyze_segments(self):
        """Perform segment analysis on the loaded score"""
        if not self.parsed_score:
            raise ValueError("No score loaded. Call load_score first.")

        if self.parsed_score == 'dir':
            self.segments = []
            self.ir_symbols = []
            combined_nmat = None
            combined_narr = []
            combined_sarr = []

            for piece in os.listdir(self.score_path):
                piece_path = os.path.join(self.score_path, piece)

                if not os.path.isfile(piece_path):
                    continue

                try:
                    parsed_score = converter.parse(piece_path)
                    nmat, narr, sarr = parse_score_elements(parsed_score)

                    # Process individual file
                    ir_symbols = assign_ir_symbols(narr)
                    self.ir_symbols.extend(ir_symbols)
                    ir_nmat = ir_symbols_to_matrix(ir_symbols, nmat)
                    segments = segmentgestalt(ir_nmat)

                    # Concatenate data from multiple files
                    if combined_nmat is None:
                        combined_nmat = nmat.copy()
                    else:
                        # Assuming nmat is a DataFrame, concat vertically
                        combined_nmat = pd.concat([combined_nmat, nmat], ignore_index=True)

                    combined_narr.extend(narr)
                    combined_sarr.extend(sarr)
                    self.segments.extend(segments)
                except Exception as e:
                    print(f"Error processing {piece_path}: {e}")
                    continue

            # Store the combined data
            self.nmat = combined_nmat
            self.narr = combined_narr
            self.sarr = combined_sarr

        else:
            nmat, narr, sarr = parse_score_elements(self.parsed_score)
            # nmat['mobility'] = mobility(nmat) # calculate mobility and add column to raw nmat
            # nmat['tessitura'] = tessitura(nmat) # calculate tessitura and add column to raw nmat
            # nmat['expectancy'] = calculate_note_expectancy_scores(nmat)

            ir_symbols = assign_ir_symbols(narr)
            self.ir_symbols = ir_symbols
            ir_nmat = ir_symbols_to_matrix(ir_symbols, nmat)
            # ir_nmat = assign_ir_pattern_indices(ir_nmat)
            self.segments = segmentgestalt(ir_nmat)

            self.nmat = nmat  # save raw nmat
            self.narr = narr
            self.sarr = sarr
        return self

    def preprocess_segments(self):
        """Preprocess the analyzed segments"""
        if self.segments is None:
            raise ValueError("No segments analyzed. Call analyze_segments first.")

        self.prepped_segments = preprocess_segments(self.segments)
        return self

    def calculate_distance_matrix(self):
        """Calculate distance matrix for preprocessed segments"""
        if self.prepped_segments is None:
            raise ValueError("No preprocessed segments. Call preprocess_segments first.")

        self.distance_matrix = segments_to_distance_matrix(self.prepped_segments)
        return self

    def save_segments(self, filepath):
        """Save segments to a pickle file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.segments, f)

    def load_segments(self, filepath):
        """Load segments from a pickle file"""
        with open(filepath, 'rb') as f:
            self.segments = pickle.load(f)
        return self

    def run(self, filepath):
        self.load_score(filepath)
        self.analyze_segments()
        self.preprocess_segments()
        self.calculate_distance_matrix()


class MusicVisualizer:
    """Handles visualization of musical segments and graphs"""

    def __init__(self, analyzer=None):
        self.analyzer = analyzer

    def visualize_colored_segments(self):
        """Create and display a score with colored segments"""
        if not self.analyzer or not self.analyzer.parsed_score:
            raise ValueError("No analyzed score available")

        colored_score = visualize_score_with_colored_segments(
            self.analyzer.parsed_score,
            self.analyzer.segments
        )
        colored_score.show()

    def visualize_multiple_segments(self, num_segments=5):
        """Display multiple segments using the MultiSegmentVisualizer"""
        if not self.analyzer or not self.analyzer.segments:
            raise ValueError("No segments available")

        MultiSegmentVisualizer(
            self.analyzer.segments,
            self.analyzer.parsed_score,
            num_segments
        )

    def visualize_ir(self):
        """Visualize the IR patterns in the piece"""
        if not self.analyzer or not self.analyzer.ir_symbols:
            raise ValueError("No IR symbols available")

        ir_score = visualize_notes_with_symbols(self.analyzer.ir_symbols, self.analyzer.parsed_score)
        ir_score.show()

    def visualize_knn_graph(self, k=3, seed=69, title=None):
        """Create and display a KNN graph of the segments"""
        # if not self.analyzer or not self.analyzer.distance_matrix is None:
        #     raise ValueError("No distance matrix available")

        title = title or "Segment Analysis"
        distance_matrix_to_knn_graph_scaled(
            k,
            self.analyzer.distance_matrix,
            f"{title}\n",
            seed
        )


class GraphBuilder:
    def __init__(self, k: int, distance_matrix: np.ndarray, segments: list[pd.DataFrame]):
        self.k = k
        self.graph = None
        self.graphs = None
        self.distance_matrix = distance_matrix
        self.segments = segments

    def construct_graph(self):
        knn_graph = kneighbors_graph(self.distance_matrix, n_neighbors=self.k, mode='connectivity')
        G = nx.from_scipy_sparse_array(knn_graph)

        # for i in range(len(self.segments)):
        #     G.nodes[i]['label'] = np.round(self.segments[i]['expectancy'].mean(), decimals=2)
        #     G.nodes[i]['label'] = i

        # if not nx.is_connected(G):
        #     print("The KNN graph is disjoint. Ensuring connectivity...")
        #
        #     components = list(nx.connected_components(G))
        #
        #     for i in range(len(components) - 1):
        #         min_dist = np.inf
        #         closest_pair = None
        #         for node1 in components[i]:
        #             for node2 in components[i + 1]:
        #                 dist = self.distance_matrix[node1, node2]
        #                 if dist < min_dist:
        #                     min_dist = dist
        #                     closest_pair = (node1, node2)
        #         G.add_edge(closest_pair[0], closest_pair[1])
        self.graph = G
        return G


class GraphBatcher:
    def __init__(self, k=5, output_dir='./batcher_output'):
        self.k = k
        self.graphs = []
        self.graph_dict = {}

        self.segments = []
        self.segment_dict = {}

        self.distance_matrices = []
        self.distmat_dict = {}

        self.processed_files = []
        self.file_manager = MusicFileManager()
        self.analyzer = MusicSegmentAnalyzer()

        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Define paths for each pickle file
        self.graphs_path = os.path.join(self.output_dir, 'graphs.pkl')
        self.segments_path = os.path.join(self.output_dir, 'segments.pkl')
        self.distmat_path = os.path.join(self.output_dir, 'distance_matrices.pkl')
        self.processed_files_path = os.path.join(self.output_dir, 'processed_files.pkl')

        # Load previous progress if it exists
        self.load_progress()

    def batch(self):
        for file in self.file_manager.files:
            # Skip files we've already processed
            if file in self.processed_files:
                print(f"Skipping already processed file: {file}")
                continue

            print(f"Analyzing {file}")
            try:
                self.analyzer.run(self.file_manager.files[file])
                builder = GraphBuilder(self.k,
                                       self.analyzer.distance_matrix,
                                       self.analyzer.prepped_segments)

                # Create the graph
                graph = builder.construct_graph()

                # Add to lists
                self.graphs.append(graph)
                self.segments.append(self.analyzer.prepped_segments)
                self.distance_matrices.append(self.analyzer.distance_matrix)
                self.processed_files.append(file)

                # Add to dictionaries with file as key
                self.graph_dict[file] = graph
                self.segment_dict[file] = self.analyzer.prepped_segments
                self.distmat_dict[file] = self.analyzer.distance_matrix

                # Save progress after each file
                self.save_progress()

            except Exception as e:
                print(f"Error parsing: {file} at {self.file_manager.files[file]}. Skipping file")
                print(traceback.format_exc())
                continue

    def save_progress(self):
        """Save each variable to its own pickle file, overwriting previous versions."""

        # Helper function for safe saving
        def safe_save(data, path):
            temp_path = f"{path}.tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
            if os.path.exists(path):
                os.remove(path)
            os.rename(temp_path, path)

        # Save each dictionary to its own file
        safe_save(self.graph_dict, self.graphs_path)
        safe_save(self.segment_dict, self.segments_path)
        safe_save(self.distmat_dict, self.distmat_path)
        safe_save(self.processed_files, self.processed_files_path)

        print(f"Progress saved: {len(self.processed_files)} files processed")

    def load_progress(self):
        """Load previous progress from individual pickle files if available."""

        # Helper function for safe loading
        def safe_load(path, default_value):
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            return default_value

        # Load each dictionary from its own file
        self.graph_dict = safe_load(self.graphs_path, {})
        self.segment_dict = safe_load(self.segments_path, {})
        self.distmat_dict = safe_load(self.distmat_path, {})
        self.processed_files = safe_load(self.processed_files_path, [])

        # Rebuild the lists from the dictionaries
        if self.processed_files:
            self.graphs = [self.graph_dict[f] for f in self.processed_files]
            self.segments = [self.segment_dict[f] for f in self.processed_files]
            self.distance_matrices = [self.distmat_dict[f] for f in self.processed_files]

            print(f"Loaded previous progress: {len(self.processed_files)} files already processed")
