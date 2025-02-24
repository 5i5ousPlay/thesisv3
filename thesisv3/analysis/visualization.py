import copy
import os
import tempfile

import ipywidgets as widgets
import pandas as pd
from IPython.display import display, clear_output
from music21 import (
    stream,
    note,
    chord,
    expressions,
    layout,
    environment
)
from pygame import mixer


def visualize_segment(segments, segment_index, original_score, show_score=True):
    """
    Visualizes a specific segment from a musical score using an additive approach,
    building a new score with only the desired elements.

    Parameters:
        segments (list[pd.DataFrame]): List of DataFrames containing segment information.
            Each DataFrame should have columns:
            - onset_beats: The timing of each note
            - midi_pitch: The MIDI pitch number (0 for rests)
            - ir_symbol: (optional) The intervallic relationship symbol
        segment_index (int): Index of the segment to visualize
        original_score (music21.stream.Score): The original score to maintain structure
        show_score (bool): Whether to automatically display the score (default: True)

    Returns:
        music21.stream.Score: The modified score showing only the segment notes

    Raises:
        ValueError: If segment_index is invalid
    """
    # Define the color map for IR symbols
    color_map = {
        'P': 'blue',  # IR1: P (Process)
        'D': 'green',  # IR2: D (Duplication)
        'IP': 'red',  # IR3: IP (Intervallic Process)
        'ID': 'orange',  # IR4: ID (Intervallic Duplication)
        'VP': 'purple',  # IR5: VP (Vector Process)
        'R': 'cyan',  # IR6: R (Reversal)
        'IR': 'magenta',  # IR7: IR (Intervallic Reversal)
        'VR': 'yellow',  # IR8: VR (Vector Reversal)
        'M': 'pink',  # IR9: M (Monad)
        'd': 'lime',  # IR10 d (Dyad)
        'rest': 'black'  # Default color for rests
    }

    if not isinstance(segments, list) or segment_index >= len(segments):
        raise ValueError(f"Invalid segment index. Must be between 0 and {len(segments) - 1}")

    # Make a deep copy of the original score to preserve its structure
    score = copy.deepcopy(original_score)

    # Create a completely new score
    new_score = stream.Score()

    # First check if we have any staff groups (like piano's grand staff)
    part_groups = []
    for element in score.elements:
        if isinstance(element, layout.StaffGroup):
            part_groups.append(copy.deepcopy(element))

    # Add any staff groups to the new score first
    for group in part_groups:
        new_score.append(group)

    # Copy parts while preserving their relationships
    for part in score.parts:
        new_part = stream.Part()

        # Preserve staff layout information
        for elem in part.getElementsByClass('LayoutClass'):
            new_part.append(copy.deepcopy(elem))

        # Copy measures and their contents
        for measure in part.getElementsByClass('Measure'):
            new_measure = stream.Measure()
            new_measure.number = measure.number

            # Copy time signature if it exists
            ts = measure.timeSignature
            if ts:
                new_measure.timeSignature = copy.deepcopy(ts)
            # Copy clef if it exists
            cl = measure.clef
            if cl:
                new_measure.clef = copy.deepcopy(cl)
            # Copy key signature if it exists
            ks = measure.keySignature
            if ks:
                new_measure.keySignature = copy.deepcopy(ks)

            # Copy notes, rests, and chords
            for elem in measure.getElementsByClass(['Note', 'Rest', 'Chord']):
                new_measure.append(copy.deepcopy(elem))

            new_part.append(new_measure)

        # Preserve any staff identifiers or group memberships
        if hasattr(part, 'staffName'):
            new_part.staffName = part.staffName
        if hasattr(part, 'groupName'):
            new_part.groupName = part.groupName

        new_score.append(new_part)

    # Add segment number as a text expression at the top
    segment_text = expressions.TextExpression(f'Segment {segment_index}')
    segment_text.style.absoluteY = 75  # Position above the staff
    segment_text.style.fontSize = 14
    first_measure = new_score.parts[0].getElementsByClass('Measure')[0]
    first_measure.insert(0, segment_text)

    # Get the selected segment
    segment = segments[segment_index]

    # Create a set of onsets in the segment for quick lookup
    segment_onsets = set(segment['onset_beats'].values)

    # Iterate over the parts of the new_score
    for part in new_score.parts:
        measures_to_keep = []
        # First pass: identify measures with segment notes and transform notes
        for measure in part.getElementsByClass(stream.Measure):
            has_segment_notes = False
            for element in measure:
                if isinstance(element, (note.Note, note.Rest, chord.Chord)):
                    element_offset = element.getOffsetInHierarchy(new_score)

                    if element_offset in segment_onsets:
                        has_segment_notes = True
                        # Find the corresponding row in our segment DataFrame
                        segment_row = segment[segment['onset_beats'] == element_offset].iloc[0]

                        # If it's a rest in our segment data
                        if segment_row['midi_pitch'] == 0:
                            new_element = note.Rest(quarterLength=element.duration.quarterLength)
                            new_element.style.color = color_map['rest']
                        else:
                            # Create a new note with the pitch from our segment
                            new_element = note.Note(
                                pitch=segment_row['midi_pitch'],
                                quarterLength=element.duration.quarterLength
                            )

                            # Add IR symbol and color if available
                            if 'ir_symbol' in segment_row:
                                ir_symbol = segment_row['ir_symbol']
                                new_element.lyric = ir_symbol
                                if ir_symbol in color_map:
                                    new_element.style.color = color_map[ir_symbol]

                        # Replace the element in the measure
                        measure.replace(element, new_element)
                    else:
                        # Remove notes not in our segment
                        measure.remove(element)

            if has_segment_notes:
                measures_to_keep.append(measure)

        # Clear all measures from the part
        for measure in part.getElementsByClass(stream.Measure):
            part.remove(measure)

        # Add back only the measures we want to keep
        for measure in measures_to_keep:
            part.append(measure)

        # If the part is empty after removing measures, remove it from the score
        if len(part.getElementsByClass(stream.Measure)) == 0:
            new_score.remove(part)

    # Show the updated score if requested
    if show_score:
        new_score.show()

    return new_score


def visualize_segment_deprecated(segments, segment_index, original_score, show_score=True):
    """
    Legacy version of visualize_segment that uses a subtractive approach,
    removing unwanted elements from a copied score.

    Parameters:
        segments (list[pd.DataFrame]): List of DataFrames containing segment information.
            Each DataFrame should have columns:
            - onset_beats: The timing of each note
            - midi_pitch: The MIDI pitch number (0 for rests)
            - ir_symbol: (optional) The intervallic relationship symbol
        segment_index (int): Index of the segment to visualize
        original_score (music21.stream.Score): The original score to maintain structure
        show_score (bool): Whether to automatically display the score (default: True)

    Returns:
        music21.stream.Score: The modified score showing only the segment notes

    Raises:
        ValueError: If segment_index is invalid

    Note:
        This is a deprecated version kept for reference. Use visualize_segment() instead.
    """
    # Define the color map for IR symbols
    color_map = {
        'P': 'blue',  # IR1: P (Process)
        'D': 'green',  # IR2: D (Duplication)
        'IP': 'red',  # IR3: IP (Intervallic Process)
        'ID': 'orange',  # IR4: ID (Intervallic Duplication)
        'VP': 'purple',  # IR5: VP (Vector Process)
        'R': 'cyan',  # IR6: R (Reversal)
        'IR': 'magenta',  # IR7: IR (Intervallic Reversal)
        'VR': 'yellow',  # IR8: VR (Vector Reversal)
        'M': 'pink',  # IR9: M (Monad)
        'd': 'lime',  # IR10 d (Dyad)
        'rest': 'black'  # Default color for rests
    }

    if not isinstance(segments, list) or segment_index >= len(segments):
        raise ValueError(f"Invalid segment index. Must be between 0 and {len(segments) - 1}")

    # Make a deep copy of the original score to preserve its structure
    new_score = copy.deepcopy(original_score)

    # Remove all TextBox elements and TextExpressions from the score
    for part in new_score.parts:
        for measure in part.recurse().getElementsByClass('Measure'):
            for elem in measure.getElementsByClass(['TextExpression', 'TextBox']):
                measure.remove(elem)

        # Also check for text elements directly in the part
        for elem in part.getElementsByClass(['TextExpression', 'TextBox']):
            part.remove(elem)

    # Check for any text elements in the score itself
    for elem in new_score.getElementsByClass(['TextExpression', 'TextBox']):
        new_score.remove(elem)

    # Remove all TextBox elements and TextExpressions from the score
    for elem in new_score.recurse().getElementsByClass(['TextExpression', 'TextBox']):
        elem.activeSite.remove(elem)

    # Remove metadata (including filename/title)
    new_score.metadata = None

    # Add segment number as a text expression at the top
    segment_text = expressions.TextExpression(f'Segment {segment_index}')
    segment_text.style.absoluteY = 75  # Position above the staff
    segment_text.style.fontSize = 14
    first_measure = new_score.parts[0].getElementsByClass('Measure')[0]
    first_measure.insert(0, segment_text)

    # Get the selected segment
    segment = segments[segment_index]

    # Create a set of onsets in the segment for quick lookup
    segment_onsets = set(segment['onset_beats'].values)

    # Iterate over the parts of the new_score
    for part in new_score.parts:
        measures_to_keep = []
        # First pass: identify measures with segment notes and transform notes
        for measure in part.getElementsByClass(stream.Measure):
            has_segment_notes = False
            for element in measure:
                if isinstance(element, (note.Note, note.Rest, chord.Chord)):
                    element_offset = element.getOffsetInHierarchy(new_score)

                    if element_offset in segment_onsets:
                        has_segment_notes = True
                        # Find the corresponding row in our segment DataFrame
                        segment_row = segment[segment['onset_beats'] == element_offset].iloc[0]

                        # If it's a rest in our segment data
                        if segment_row['midi_pitch'] == 0:
                            new_element = note.Rest(quarterLength=element.duration.quarterLength)
                            new_element.style.color = color_map['rest']
                        else:
                            # Create a new note with the pitch from our segment
                            new_element = note.Note(
                                pitch=segment_row['midi_pitch'],
                                quarterLength=element.duration.quarterLength
                            )

                            # Add IR symbol and color if available
                            if 'ir_symbol' in segment_row:
                                ir_symbol = segment_row['ir_symbol']
                                new_element.lyric = ir_symbol
                                if ir_symbol in color_map:
                                    new_element.style.color = color_map[ir_symbol]

                        # Replace the element in the measure
                        measure.replace(element, new_element)
                    else:
                        # Remove notes not in our segment
                        measure.remove(element)

            if has_segment_notes:
                measures_to_keep.append(measure)

        # Clear all measures from the part
        for measure in part.getElementsByClass(stream.Measure):
            part.remove(measure)

        # Add back only the measures we want to keep
        for measure in measures_to_keep:
            part.append(measure)

        # If the part is empty after removing measures, remove it from the score
        if len(part.getElementsByClass(stream.Measure)) == 0:
            new_score.remove(part)

    # Show the updated score if requested
    if show_score:
        new_score.show()

    return new_score


class MultiSegmentVisualizer:
    def __init__(self, segments, original_score, max_segments=4):
        self.segments = segments
        self.original_score = original_score
        self.max_segments = max_segments
        self.active_segments = 1
        self.segment_widgets = []
        self.current_scores = [None] * max_segments
        self.temp_midi_files = []  # Keep track of temporary files

        # Initialize pygame mixer for audio playback
        mixer.init()

        # Setup environment for music21
        self.env = environment.Environment()

        self.setup_widgets()

    def __del__(self):
        """Cleanup temporary files when the visualizer is destroyed"""
        for temp_file in self.temp_midi_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def create_segment_controls(self, index):
        """Create a set of controls for a single segment"""
        segment_box = widgets.VBox([
            widgets.HBox([
                widgets.Dropdown(
                    options=[(f'Segment {i}', i) for i in range(len(self.segments))],
                    value=index,
                    description=f'Segment {index + 1}:',
                    style={'description_width': 'initial'},
                    layout=widgets.Layout(width='250px')
                ),
                widgets.Button(
                    description=f'Play Segment {index + 1}',
                    button_style='success',
                    layout=widgets.Layout(width='150px')
                ),
                widgets.Button(
                    description='Stop',
                    button_style='danger',
                    layout=widgets.Layout(width='100px')
                )
            ]),
            widgets.HTML(
                value=f'<p>Select segment {index + 1} to view details</p>',
                layout=widgets.Layout(width='100%', padding='10px')
            ),
            widgets.Output()
        ])

        # Add play button callback
        segment_box.children[0].children[1].on_click(
            lambda b, idx=index: self.play_segment(idx)
        )

        # Add stop button callback
        segment_box.children[0].children[2].on_click(
            lambda b: self.stop_playback()
        )

        return segment_box

    def setup_widgets(self):
        # Create add/remove segment controls
        self.add_segment_btn = widgets.Button(
            description='Add Segment',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )

        self.remove_segment_btn = widgets.Button(
            description='Remove Segment',
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )

        self.update_button = widgets.Button(
            description='Update All',
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )

        # Add callbacks
        self.add_segment_btn.on_click(self.add_segment)
        self.remove_segment_btn.on_click(self.remove_segment)
        self.update_button.on_click(self.update_visualization)

        # Create first segment controls
        self.segment_widgets.append(self.create_segment_controls(0))

        # Create control panel
        self.controls = widgets.HBox([
            self.add_segment_btn,
            self.remove_segment_btn,
            self.update_button
        ])

        # Layout all widgets
        self.widget_box = widgets.VBox([
            self.controls,
            widgets.VBox(self.segment_widgets)
        ])

        # Display the interface
        display(self.widget_box)

    def stop_playback(self):
        """Stop any currently playing audio"""
        mixer.music.stop()

    def play_segment(self, index):
        """Play audio for the specified segment"""
        if self.current_scores[index]:
            try:
                # Create a temporary file for the MIDI
                with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                    temp_midi_path = tmp.name
                    self.temp_midi_files.append(temp_midi_path)

                # Save the score as MIDI
                self.current_scores[index].write('midi', fp=temp_midi_path)

                # Stop any currently playing audio
                mixer.music.stop()

                # Load and play the new MIDI file
                mixer.music.load(temp_midi_path)
                mixer.music.play()

            except Exception as e:
                print(f"Error playing segment: {str(e)}")

    def add_segment(self, _):
        """Add a new segment comparison if under max limit"""
        if self.active_segments < self.max_segments:
            new_segment = self.create_segment_controls(self.active_segments)
            self.segment_widgets.append(new_segment)
            self.active_segments += 1
            self.widget_box.children = (
                self.controls,
                widgets.VBox(self.segment_widgets)
            )

    def remove_segment(self, _):
        """Remove the last segment comparison if more than one exists"""
        if self.active_segments > 1:
            removed_segment = self.segment_widgets.pop()
            self.active_segments -= 1
            self.current_scores[self.active_segments] = None
            self.widget_box.children = (
                self.controls,
                widgets.VBox(self.segment_widgets)
            )

    def get_segment_info(self, segment_idx):
        """Generate HTML info for a segment"""
        segment_df = self.segments[segment_idx]
        return f"""
        <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>
            <h4>Segment {segment_idx} Details:</h4>
            <ul>
                <li>Number of notes: {len(segment_df)}</li>
                <li>Duration (beats): {segment_df['onset_beats'].max() - segment_df['onset_beats'].min():.2f}</li>
            </ul>
        </div>
        """

    def update_visualization(self, _=None):
        """Update all active segment visualizations"""
        for idx, segment_box in enumerate(self.segment_widgets):
            # Get the selected segment index from the dropdown
            selected_segment = segment_box.children[0].children[0].value

            # Update info
            segment_box.children[1].value = self.get_segment_info(selected_segment)

            # Update visualization
            with segment_box.children[2]:
                clear_output(wait=True)
                self.current_scores[idx] = visualize_segment(
                    self.segments,
                    selected_segment,
                    self.original_score,
                    show_score=False
                )
                self.current_scores[idx].show()


def get_segment_colors():
    """
    Returns a list of distinct colors for segments.
    Colors are chosen to be visually distinguishable and music-notation friendly.
    """
    return [
        '#1f77b4',  # blue
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#17becf',  # cyan
        '#bcbd22',  # olive
        '#ff7f0e',  # orange
        '#7f7f7f',  # gray
    ]


def analyze_segment(segment):
    """
    Analyzes a segment to extract key characteristics.

    Parameters:
    segment (pd.DataFrame): A segment DataFrame

    Returns:
    dict: Dictionary containing segment characteristics
    """
    # Count IR symbols
    ir_symbols = segment['ir_symbol'].value_counts()
    dominant_pattern = ir_symbols.index[0] if not ir_symbols.empty else 'None'

    # Calculate duration
    total_duration = segment['duration_beats'].sum()

    return {
        'total_notes': len(segment),
        'duration': total_duration,
        'dominant_pattern': dominant_pattern,
        'ir_distribution': dict(ir_symbols)
    }


def create_segment_info_string(segment_analysis):
    """
    Creates a formatted string of segment information.

    Parameters:
    segment_analysis (dict): Dictionary of segment characteristics

    Returns:
    str: Formatted information string
    """
    info = [
        f"Notes: {segment_analysis['total_notes']}",
        f"Duration: {segment_analysis['duration']:.1f} beats",
        f"Dominant Pattern: {segment_analysis['dominant_pattern']}",
        "\nIR Distribution:"
    ]

    for pattern, count in segment_analysis['ir_distribution'].items():
        info.append(f"  {pattern}: {count}")

    return "\n".join(info)


def visualize_score_with_colored_segments(original_score, segments):
    """
    Creates a visualization of the full score with color-coded segments and labels.

    Parameters:
    original_score (music21.stream.Score): The original score
    segments (list): List of segment DataFrames

    Returns:
    music21.stream.Score: Score with colored segments and annotations
    """
    # Make a deep copy of the original score
    new_score = copy.deepcopy(original_score)

    # Get colors for segments
    colors = get_segment_colors()

    # For each segment, color its notes and add labeled boundaries
    for i, segment in enumerate(segments):
        # Get color for this segment (cycle through colors if more segments than colors)
        segment_color = colors[i % len(colors)]

        # Get characteristics for this segment
        analysis = analyze_segment(segment)

        # Create label text
        label_text = f"[Segment {i + 1}]\n{analysis['dominant_pattern']}"

        # Create an expression for the label
        label = expressions.TextExpression(label_text)
        label.style.fontSize = 12
        label.style.color = segment_color
        label.placement = 'above'

        # Get the segment's start and end times
        segment_start = segment['onset_beats'].iloc[0]
        segment_end = segment['onset_beats'].iloc[-1] + segment['duration_beats'].iloc[-1]

        # Create a set of segment onsets for quick lookup
        segment_onsets = set(segment['onset_beats'].values)

        # Add the label and color the notes
        for part in new_score.parts:
            # Add label at segment start
            for measure in part.getElementsByClass(stream.Measure):
                measure_offset = measure.getOffsetInHierarchy(new_score)
                if measure_offset <= segment_start < measure_offset + measure.duration.quarterLength:
                    # Add label to the measure
                    label.offset = segment_start - measure_offset
                    measure.insert(0, label)

                # Color notes in this segment
                for element in measure:
                    if isinstance(element, (note.Note, chord.Chord)):
                        element_offset = element.getOffsetInHierarchy(new_score)
                        # Check if this element is in our segment
                        if element_offset in segment_onsets:
                            element.style.color = segment_color

    return new_score


def visualize_notes_with_symbols(notes_with_symbols, original_score):
    """
    Visualizes notes with their assigned IR symbols and colors in a music21 score.

    Parameters:
    notes_with_symbols (list): A list of tuples containing each element, its IR symbol, and its color.
    original_score (music21.stream.Score): The original music21 score to replicate structural attributes.

    Returns:
    None
    """
    # Make a deep copy of the original score to preserve its structure
    new_score = copy.deepcopy(original_score)

    # Flatten notes_with_symbols for easy indexing
    symbols_iter = iter(notes_with_symbols)

    # Iterate over the parts of the new_score
    for part in new_score.parts:
        # Iterate over measures in the part
        for measure in part.getElementsByClass(stream.Measure):
            # Iterate over elements in the measure
            for element in measure:
                if isinstance(element, (note.Note, note.Rest, chord.Chord)):
                    try:
                        symbol_element, symbol, color = next(symbols_iter)
                        # Apply color and lyric if the elements match
                        if element == symbol_element:
                            element.style.color = color
                            element.lyric = symbol
                    except StopIteration:
                        break  # No more symbols to assign

    return new_score
