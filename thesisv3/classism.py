import pickle

import music21
from music21 import converter, environment

from thesisv3.analysis.visualization import *
from thesisv3.preprocessing.preprocessing import *
from thesisv3.building.building import distance_matrix_to_knn_graph_scaled
from thesisv3.utils.file_manager import MusicFileManager

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
        self.nmat = None # create field for raw nmat

    def load_score(self, score_path=None):
        """Load and parse a music score"""
        if score_path:
            self.score_path = score_path
        if not self.score_path:
            raise ValueError("No score path provided")

        self.parsed_score = converter.parse(self.score_path)
        return self

    def analyze_segments(self):
        """Perform segment analysis on the loaded score"""
        if not self.parsed_score:
            raise ValueError("No score loaded. Call load_score first.")

        nmat, narr, sarr = parse_score_elements(self.parsed_score)
        nmat['mobility'] = mobility(nmat) # calculate mobility and add column to raw nmat
        self.nmat = nmat # save raw nmat
        ir_symbols = assign_ir_symbols(narr)
        ir_nmat = ir_symbols_to_matrix(ir_symbols, nmat)
        ir_nmat = assign_ir_pattern_indices(ir_nmat)
        self.segments = segmentgestalt(ir_nmat)
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
