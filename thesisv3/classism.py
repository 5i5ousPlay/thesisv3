import pickle
import traceback

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
        self.nmat = None  # create field for raw nmat
        self.narr = None
        self.sarr = None
        self.ir_symbols = None

    def load_score(self, score_path=None):
        def custom_chordify(score, tolerance=0.001):
            """
            Creates a new score where simultaneous notes across voices
            (within a given time tolerance) are merged into chords.
            Single notes remain as note.Note objects.

            Parameters:
                score (music21.stream.Score): The original score.
                tolerance (float): Tolerance for grouping onsets (in quarter lengths).

            Returns:
                music21.stream.Score: A new score with merged simultaneous notes.
            """
            # Clone the score so we don't modify the original
            new_score = copy.deepcopy(score)

            # Determine if new_score has parts; if not, treat it as a single part.
            parts_to_process = new_score.parts if hasattr(new_score, 'parts') else [new_score]

            # Process each part separately
            for part in parts_to_process:
                # Process each measure individually to preserve structure
                for meas in part.getElementsByClass('Measure'):
                    # Get all Note and Rest objects in this measure (from all voices)
                    events = meas.recurse().getElementsByClass([note.Note, note.Rest])
                    # Group events by their offset within the measure
                    offset_groups = {}
                    for ev in events:
                        # Round the offset to the nearest tolerance unit.
                        off_key = round(ev.offset / tolerance) * tolerance
                        offset_groups.setdefault(off_key, []).append(ev)

                    # For each offset group, if there are two or more notes, merge them.
                    for off, group in offset_groups.items():
                        # Filter out rests – usually you want to chordify only the sounding pitches.
                        note_group = [n for n in group if isinstance(n, note.Note)]
                        if len(note_group) >= 2:
                            # Create a chord from the grouped notes.
                            new_chord = chord.Chord(note_group)
                            # Choose a duration (for example, take the duration of the first note)
                            new_chord.duration = note_group[0].duration
                            # Optionally, you might transfer other attributes (e.g., tie, articulation)

                            # Remove the original notes from their parent containers.
                            for n in note_group:
                                if n.activeSite is not None:
                                    n.activeSite.remove(n)
                            # Insert the chord at the given offset into the measure.
                            meas.insert(off, new_chord)

            return new_score

        def unchordify_singletons(chordified_score):
            for c in chordified_score.recurse().getElementsByClass(chord.Chord):
                # If chord has only one pitch, replace it with a Note.
                if len(c.pitches) == 1:
                    n = note.Note(c.pitches[0])
                    n.duration = c.duration
                    n.offset = c.offset
                    # Replace the chord with the note in its parent container.
                    c.activeSite.replace(c, n)
            return chordified_score

        """Load and parse a music score"""
        if score_path:
            self.score_path = score_path
        if not self.score_path:
            raise ValueError("No score path provided")

        self.parsed_score = converter.parse(self.score_path)
        # self.parsed_score = self.parsed_score.makeNotation()
        # self.parsed_score = self.parsed_score.makeMeasures()
        # self.parsed_score = custom_chordify(self.parsed_score)
        # self.parsed_score = self.parsed_score.chordify()
        # self.parsed_score = unchordify_singletons(self.parsed_score)
        return self



    def analyze_segments(self):
        """Perform segment analysis on the loaded score"""
        if not self.parsed_score:
            raise ValueError("No score loaded. Call load_score first.")

        nmat, narr, sarr = parse_score_elements(self.parsed_score)
        nmat['mobility'] = mobility(nmat)  # calculate mobility and add column to raw nmat
        nmat['tessitura'] = tessitura(nmat)  # calculate tessitura and add column to raw nmat
        nmat['expectancy'] = calculate_note_expectancy_scores(nmat)
        self.nmat = nmat  # save raw nmat
        self.narr = narr
        self.sarr = sarr
        ir_symbols = assign_ir_symbols(narr)
        self.ir_symbols = ir_symbols
        ir_nmat = ir_symbols_to_matrix(ir_symbols, nmat)
        # ir_nmat = assign_ir_pattern_indices(ir_nmat)
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

        for i in range(len(self.segments)):
            G.nodes[i]['label'] = np.round(self.segments[i]['expectancy'].mean(), decimals=2)

        if not nx.is_connected(G):
            print("The KNN graph is disjoint. Ensuring connectivity...")

            components = list(nx.connected_components(G))

            for i in range(len(components) - 1):
                min_dist = np.inf
                closest_pair = None
                for node1 in components[i]:
                    for node2 in components[i + 1]:
                        dist = self.distance_matrix[node1, node2]
                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = (node1, node2)
                G.add_edge(closest_pair[0], closest_pair[1])
        self.graph = G
        return G


class GraphBatcher:
    def __init__(self, k=5):
        self.k = k
        self.graphs = []
        self.segments = []
        self.distance_matrices = []
        self.processed_files = []
        self.file_manager = MusicFileManager()
        self.analyzer = MusicSegmentAnalyzer()

    def batch(self):
        for file in self.file_manager.files:
            print(f"Analyzing {file}")
            try:
                self.analyzer.run(self.file_manager.files[file])
                builder = GraphBuilder(self.k,
                                       self.analyzer.distance_matrix,
                                       self.analyzer.prepped_segments)
                self.graphs.append(builder.construct_graph())
                self.segments.append(self.analyzer.prepped_segments)
                self.distance_matrices.append(self.analyzer.distance_matrix)
                self.processed_files.append(file)

            except Exception as e:
                print(f"Error parsing: {file} at {self.file_manager.files[file]}. Skipping file")
                print(traceback.format_exc())
                continue
