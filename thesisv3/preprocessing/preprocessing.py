import re
from multiprocessing import cpu_count, Manager, Pool

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from thesisv3.utils import worker
from music21 import chord, note, stream, meter
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler


# # Usage
# Current flow: \
# converter.parse() output -> parse_score_elements() ->
# assign_ir_symbols -> ir_symbols_to_matrix ->
# assign_ir_pattern_indices -> segmentgestalt

def extract_score_elements(score):
    """
    Extracts elements from a music21 score object and organizes them into a DataFrame.

    Parameters:
    score (music21.stream.Score): The music21 score object to extract elements from.

    Returns:
    pd.DataFrame: A DataFrame containing part index, offset, duration, type, and pitch information for each element.
    """
    score = score.expandRepeats()

    elements = []

    for part_index, part in enumerate(score.parts):
        for element in part.flatten():
            element_info = {
                'part_index': part_index,
                'offset': element.offset,
                'duration': element.duration.quarterLength,
                'type': type(element).__name__
            }

            if isinstance(element, chord.Chord):
                element_info['pitches'] = [p.midi for p in element.pitches]
            elif isinstance(element, note.Rest):
                element_info['pitch'] = 0  # Representing rest with 0 pitch
            elif isinstance(element, note.Note):
                element_info['pitch'] = element.pitch.midi
            elif isinstance(element, meter.TimeSignature):
                element_info['numerator'] = element.numerator
                element_info['denominator'] = element.denominator
            else:
                continue  # Skip other element types for simplicity

            elements.append(element_info)

    elements_df = pd.DataFrame(elements)
    return elements_df


def recreate_score(elements_df):
    """
    Recreates a music21 score object from a DataFrame of score elements.

    Parameters: elements_df (pd.DataFrame): A DataFrame containing part index, offset, duration, type, and pitch
    information for each element.

    Returns:
    music21.stream.Score: The recreated music21 score object.
    """
    score = stream.Score()
    parts_dict = {}

    for _, row in elements_df.iterrows():
        part_index = row['part_index']
        if part_index not in parts_dict:
            parts_dict[part_index] = stream.Part()

        element_type = row['type']
        offset = row['offset']
        duration = row['duration']

        if element_type == 'Chord':
            pitches = [note.Note(p) for p in row['pitches']]
            element = chord.Chord(pitches, quarterLength=duration)
        elif element_type == 'Rest':
            element = note.Rest(quarterLength=duration)
        elif element_type == 'Note':
            element = note.Note(row['pitch'], quarterLength=duration)
        elif element_type == 'TimeSignature':
            element = meter.TimeSignature(f"{row['numerator']}/{row['denominator']}")
        else:
            continue

        element.offset = offset
        parts_dict[part_index].append(element)

    for part in parts_dict.values():
        score.append(part)

    return score


def parse_score_elements(score: stream.Score, all_parts: bool = False) -> tuple[pd.DataFrame, list, list]:
    """
    Parses a music21 score object into a DataFrame of note attributes and lists of note and chord elements.
    By default, only processes the first part unless all_parts=True.

    Parameters:
    score (music21.stream.Score): The music21 score object to parse.
    all_parts (bool): If True, process all parts. If False, only process the first part. Defaults to False.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: A DataFrame with onset (global and relative to measure), duration, MIDI pitch,
                        pitch class, octave, and beat strength for each note.
        - list: A list of note and chord elements.
        - list: A list of all elements processed.
    """
    trashed_elements = 0
    narr = []  # List for successfully processed note/chord/rest elements
    sarr = []  # List for all elements encountered
    nmat = pd.DataFrame(columns=[
        'onset_beats',  # Global onset in beats for the whole piece
        'onset_beats_in_measure',  # Onset relative to the measure
        'duration_beats',
        'midi_pitch',
        'pitch_class',
        'octave',
        'beat_strength'
    ])

    onset_beat = 0
    parts_to_process = score.parts if all_parts else [score.parts[0]]

    # Helper function to process a single musical element (note, chord, or rest)
    def process_element(e, current_onset):
        duration = e.duration.quarterLength
        beat_strength = getattr(e, 'beatStrength', None)
        # Use the element's own offset for the onset within its measure
        onset_in_measure = e.offset
        if isinstance(e, chord.Chord):
            root = e.root()
            midi_pitch = root.midi
            pitch_class = root.pitchClass
            octave = root.octave
        elif isinstance(e, note.Note):
            midi_pitch = e.pitch.midi
            pitch_class = e.pitch.pitchClass
            octave = e.pitch.octave
        elif isinstance(e, note.Rest):
            midi_pitch = 0
            pitch_class = 0
            octave = 0
        else:
            return None, 0
        row = [current_onset, onset_in_measure, duration, midi_pitch, pitch_class, octave, beat_strength]
        return row, duration

    for part in parts_to_process:
        for measure in part.getElementsByClass(stream.Measure):
            # Process only the first Voice in the measure (if any)
            voice_processed = False
            for element in measure:
                sarr.append(element)
                if isinstance(element, stream.Voice):
                    # Process only if the voice's id is "1" or if no id is set and no voice has been processed yet
                    if (element.id is not None and element.id != '1') or voice_processed:
                        continue
                    voice_processed = True
                    # Process all valid subelements in the voice
                    for subelement in element.flatten().getElementsByClass([note.Note, chord.Chord, note.Rest]):
                        row, dur = process_element(subelement, onset_beat)
                        if row is not None:
                            nmat.loc[len(nmat)] = row
                            narr.append(subelement)
                        else:
                            trashed_elements += 1
                        onset_beat += dur
                else:
                    row, dur = process_element(element, onset_beat)
                    if row is not None:
                        nmat.loc[len(nmat)] = row
                        narr.append(element)
                    else:
                        trashed_elements += 1
                    onset_beat += dur

    return nmat, narr, sarr


def calculate_ir_symbol(interval1, interval2, threshold=5):
    """
    Calculates the Implication-Realization symbol based on the intervals between notes.

    Parameters:
    interval1 (int): The interval between the first and second notes.
    interval2 (int): The interval between the second and third notes.
    threshold (int): The threshold value for determining the type of relationship.

    Returns:
    str: The IR symbol representing the relationship between the intervals.
    """
    direction = interval1 * interval2
    abs_difference = abs(interval2 - interval1)

    if direction > 0 and abs_difference < threshold:
        return 'P'  # Process
    elif interval1 == interval2 == 0:
        return 'D'  # Duplication
    elif (interval1 * interval2 < 0) and (-threshold <= abs_difference <= threshold) and (
            abs(interval2) != abs(interval1)):
        return 'IP'  # Intervallic Process
    elif (interval1 * interval2 < 0) and (abs(interval2) == abs(interval1)):
        return 'ID'  # Intervallic Duplication
    elif (direction > 0) and (abs_difference >= threshold) and (abs(interval1) <= threshold):
        return 'VP'  # Vector Process
    elif (interval1 * interval2 < 0) and (abs_difference >= threshold) and (abs(interval1) >= threshold):
        return 'R'  # Reversal
    elif (direction > 0) and (abs_difference >= threshold) and (abs(interval1) >= threshold):
        return 'IR'  # Intervallic Reversal
    elif (interval1 * interval2 < 0) and (abs_difference >= threshold) and (abs(interval1) <= threshold):
        return 'VR'  # Vector Reversal
    elif interval2 == 0 and not (interval1 < -5 or interval1 > 5):
        return 'IP'
    elif interval2 == 0 and (interval1 < -5 or interval1 > 5):
        return 'R'
    elif interval1 == 0 and not (interval2 < -5 or interval2 > 5):
        return 'P'
    elif interval1 == 0 and (interval2 < -5 or interval2 > 5):
        return 'VR'
    else:
        return 'M'  # Default to Monad if none of the above


def assign_ir_symbols_old(note_array):
    """
    Assigns IR symbols and colors to each element in the score array.

    Parameters:
    score_array (list): A list of music21 note and chord elements.

    Returns:
    list: A list of tuples containing each element, its IR symbol, and its color.
    """
    symbols = []
    current_group = []
    group_pitches = []

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
    }

    def evaluate_current_group():
        if len(current_group) == 3:
            interval1 = group_pitches[1] - group_pitches[0]
            interval2 = group_pitches[2] - group_pitches[1]
            symbol = calculate_ir_symbol(interval1, interval2)
            color = color_map.get(symbol, 'black')  # Default to black if symbol is not predefined
            symbols.extend([(note, symbol, color) for note in current_group])
        elif len(current_group) == 2:
            symbols.extend([(note, 'd', color_map['d']) for note in current_group])  # Dyad
        elif len(current_group) == 1:
            symbols.extend([(note, 'M', color_map['M']) for note in current_group])  # Monad
        current_group.clear()
        group_pitches.clear()

    for element in note_array:
        if isinstance(element, note.Note):
            current_group.append(element)
            group_pitches.append(element.pitch.ps)
            if len(current_group) == 3:
                evaluate_current_group()
        elif isinstance(element, chord.Chord):
            current_group.append(element)
            group_pitches.append(element.root().ps)
            if len(current_group) == 3:
                evaluate_current_group()
        elif isinstance(element, note.Rest):
            rest_tuple = (element, 'rest', 'black')
            evaluate_current_group()
            symbols.append(rest_tuple)
        else:
            if current_group:
                evaluate_current_group()

    # Handle any remaining notes
    if current_group:
        evaluate_current_group()

    return symbols


# TODO: ir symbol assignment accounting slurred notes. Using this would require updating ir index assignment as well
def assign_ir_symbols(note_array):
    """
    Assigns IR symbols and colors to each element in the score array.

    Parameters:
    score_array (list): A list of music21 note and chord elements.

    Returns:
    list: A list of tuples containing each element, its IR symbol, and its color.
    """
    symbols = []
    current_group = []
    group_pitches = []
    last_beam_status = None

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
    }

    def evaluate_current_group():
        if len(current_group) >= 3:
            interval1 = group_pitches[1] - group_pitches[0]
            interval2 = group_pitches[2] - group_pitches[1]
            symbol = calculate_ir_symbol(interval1, interval2)
            color = color_map.get(symbol, 'black')  # Default to black if symbol is not predefined
            symbols.extend([(note, symbol, color) for note in current_group])
        elif len(current_group) == 2:
            symbols.extend([(note, 'd', color_map['d']) for note in current_group])  # Dyad
        elif len(current_group) == 1:
            symbols.extend([(note, 'M', color_map['M']) for note in current_group])  # Monad
        current_group.clear()
        group_pitches.clear()

    def get_beam_status(e):
        if not isinstance(e, (note.Note, chord.Chord)):
            return None
        beam_status = 'single'  # Default status for unbeamed notes
        if e.beams:
            # Get beam types
            beam_types = [beam.type for beam in e.beams.beamsList]
            if 'start' in beam_types:
                beam_status = 'start'
            elif 'continue' in beam_types:
                beam_status = 'continue'
            elif 'stop' in beam_types:
                beam_status = 'stop'
            elif 'partial' in beam_types:
                beam_status = 'partial'
        return beam_status

    num_notes = len(note_array)
    i = 0
    while i < num_notes:
        element = note_array[i]
        if isinstance(element, (note.Note, chord.Chord)):
            # Get beam status
            beam_status = get_beam_status(element)
            current_group.append(element)
            if isinstance(element, note.Note):
                group_pitches.append(element.pitch.ps)
            elif isinstance(element, chord.Chord):
                group_pitches.append(element.root().ps)

            # BEAM
            """fix it that the criteria for being evaluated isnt necesarily just three elements

make it so that it checks first if its currently beamed, if it is, then keep adding elements until the beam is done, and then i fbeam ends, then thats when you evaluate
if its not beamed, then follow the normal rules"""
            if (last_beam_status in ['start', 'continue', 'partial']) and (beam_status in ['continue', 'stop', 'partial']):
                if beam_status == 'stop':
                    # TODO: REVISIT IF IT IS WEIRD
                    if len(current_group) == 2:
                        i += 1
                        element2 = note_array[i]
                        if isinstance(element2, (note.Note, chord.Chord)):
                            # Get beam status
                            beam_status = get_beam_status(element2)
                            current_group.append(element2)
                            if isinstance(element2, note.Note):
                                group_pitches.append(element2.pitch.ps)
                            elif isinstance(element2, chord.Chord):
                                group_pitches.append(element2.root().ps)
                        elif isinstance(element, note.Rest):
                            rest_tuple = (element, 'rest', 'black')
                            evaluate_current_group()
                            symbols.append(rest_tuple)
                        else:
                            if current_group:
                                evaluate_current_group()
                    last_beam_status = beam_status

            elif i < num_notes - 1 and get_beam_status(note_array[i + 1]) == 'start' and beam_status == 'single':
                evaluate_current_group()
            if len(current_group) == 3:
                evaluate_current_group()
            last_beam_status = beam_status

        elif isinstance(element, note.Rest):
            rest_tuple = (element, 'rest', 'black')
            evaluate_current_group()
            symbols.append(rest_tuple)
            last_beam_status = None
            # symbols.append("POOP")
        else:
            if current_group:
                evaluate_current_group()
            last_beam_status = None
        i += 1

    # Handle any remaining notes
    if current_group:
        evaluate_current_group()

    return symbols


def visualize_notes_with_symbols(notes_with_symbols, original_score):
    """
    Visualizes notes with their assigned IR symbols and colors in a music21 score.

    Parameters:
    notes_with_symbols (list): A list of tuples containing each element, its IR symbol, and its color.
    original_score (music21.stream.Score): The original music21 score to replicate structural attributes.

    Returns:
    None
    """
    import copy
    from music21 import note, chord, stream

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

    # Show the updated score
    new_score.show()


def ir_symbols_to_matrix(note_array, note_matrix):
    """
    Assigns IR symbols to the note matrix based on the note array.

    Parameters:
    note_array (list): A list of tuples containing note data, IR symbols, and colors.
    note_matrix (pd.DataFrame): A DataFrame containing note attributes.

    Returns:
    pd.DataFrame: The updated DataFrame with assigned IR symbols.
    """
    for pointer, (note_data, ir_symbol, color) in enumerate(note_array):
        note_matrix.at[pointer, 'ir_symbol'] = ir_symbol
    return note_matrix


def assign_ir_pattern_indices(notematrix):
    """
    Assigns pattern indices to the note matrix based on IR symbols.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes and IR symbols.

    Returns:
    pd.DataFrame: The updated DataFrame with assigned pattern indices.
    """
    pattern_index = 0
    indices = []
    i = 0
    while i < len(notematrix):
        ir_symbol = notematrix.iloc[i]['ir_symbol']
        if ir_symbol == 'd':
            indices.extend([pattern_index, pattern_index])
            i += 2
        elif ir_symbol == 'M' or ir_symbol == 'rest':
            indices.append(pattern_index)
            i += 1
        else:
            indices.extend([pattern_index, pattern_index, pattern_index])
            i += 3
        pattern_index += 1
    notematrix['pattern_index'] = indices
    return notematrix


def get_onset(notematrix: pd.DataFrame, timetype='beat'):
    """
    Retrieves the onset times from the note matrix.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes.
    timetype (str): The type of time to retrieve (default is 'beat').

    Returns:
    pd.Series: A series containing the onset times.
    """
    if timetype == 'beat':
        return notematrix['onset_beats']
    else:
        raise ValueError(f"Invalid timetype: {timetype}")


def get_pitch(notematrix: pd.DataFrame, timetype='beat'):
    if timetype == 'beat':
        return notematrix['midi_pitch']
    else:
        raise ValueError(f"Invalid timetype: {timetype}")


def get_duration(notematrix: pd.DataFrame, timetype='beat') -> pd.Series:
    """
    Retrieves the duration times from the note matrix.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes.
    timetype (str): The type of time to retrieve (default is 'beat').

    Returns:
    pd.Series: A series containing the duration times.
    """
    if timetype == 'beat':
        return notematrix['duration_beats']
    else:
        raise ValueError(f"Invalid timetype: {timetype}")


def calculate_clang_boundaries(notematrix: pd.DataFrame):
    """
    Calculates clang boundaries based on note matrix attributes.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes.

    Returns:
    tuple: A tuple containing:
        - list: A list of indices representing clang boundaries.
        - pd.Series: A series indicating clang boundaries with boolean values.
    """
    cl = 2 * (get_onset(notematrix).diff().fillna(0) + get_duration(notematrix).shift(-1).fillna(0)) + abs(
        notematrix['midi_pitch'].diff().fillna(0))
    cl = cl.infer_objects()  # Ensure correct data types
    clb = (cl.shift(-1).fillna(0) > cl) & (cl.shift(1).fillna(0) > cl)
    clind = cl.index[clb].tolist()
    return clind, clb


def calculate_segment_boundaries(notematrix, clind):
    """
    Calculates segment boundaries based on clang boundaries and note attributes.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes.
    clind (list): A list of clang boundary indices.

    Returns:
    pd.Series: A series indicating segment boundaries with boolean values.
    """
    first = [0] + clind
    last = [i - 1 for i in clind] + [len(notematrix) - 1]

    mean_pitch = []
    for i in range(len(first)):
        segment = notematrix.iloc[first[i]:last[i] + 1]
        weighted_pitch_sum = (segment['midi_pitch'] * segment['duration_beats']).sum()
        total_duration = segment['duration_beats'].sum()
        if total_duration > 0:
            mean_pitch.append(weighted_pitch_sum / total_duration)
        else:
            mean_pitch.append(0)  # Avoid division by zero by assigning 0 if total_duration is 0

    segdist = []
    for i in range(1, len(first)):
        distance = (abs(mean_pitch[i] - mean_pitch[i - 1]) +
                    notematrix.iloc[first[i]]['onset_beats'] - notematrix.iloc[last[i - 1]]['onset_beats'] +
                    notematrix.iloc[first[i]]['duration_beats'] + notematrix.iloc[last[i - 1]]['duration_beats'] +
                    2 * (notematrix.iloc[first[i]]['onset_beats'] - notematrix.iloc[last[i - 1]]['onset_beats']))
        segdist.append(distance)

    segb = [(segdist[i] > segdist[i - 1] and segdist[i] > segdist[i + 1]) for i in range(1, len(segdist) - 1)]
    segind = [clind[i] for i in range(1, len(segdist) - 1) if segb[i - 1]]

    s = pd.Series(0, index=range(len(notematrix)))
    s.iloc[segind] = 1

    return s


def adjust_segment_boundaries(notematrix, s):
    """
    Adjusts segment boundaries to ensure IR patterns are not split.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes and IR symbols.
    s (pd.Series): A series indicating initial segment boundaries with boolean values.

    Returns:
    pd.Series: The adjusted series indicating segment boundaries.
    """
    adjusted_s = s.copy()
    indices_with_ones = np.where(s == 1)[0].tolist()

    for i in indices_with_ones:
        current_pattern = notematrix.iloc[i]['pattern_index']
        ir_symbol = notematrix.iloc[i]['ir_symbol']

        if ir_symbol == 'M' or ir_symbol == 'rest':
            continue
        elif ir_symbol == 'd':
            if 0 < i < len(notematrix) - 1:
                prev_index = indices_with_ones[indices_with_ones.index(i) - 1] if indices_with_ones.index(i) > 0 else 0
                next_index = indices_with_ones[indices_with_ones.index(i) + 1] if indices_with_ones.index(i) < len(
                    indices_with_ones) - 1 else len(notematrix) - 1

                if (i - prev_index) > (next_index - i):
                    adjusted_s.iloc[i] = 0
                    adjusted_s.iloc[i + 1] = 1
                else:
                    adjusted_s.iloc[i] = 0
                    adjusted_s.iloc[i - 1] = 1
            continue

        if i > 1:
            previous_pattern1 = notematrix.iloc[i - 1]['pattern_index']
            previous_pattern2 = notematrix.iloc[i - 2]['pattern_index']

            if (current_pattern == previous_pattern1) and (current_pattern == previous_pattern2):
                continue
            elif current_pattern == previous_pattern1 and current_pattern != previous_pattern2:
                adjusted_s.iloc[i] = 0
                adjusted_s.iloc[i + 1] = 1
            elif current_pattern != previous_pattern1 and current_pattern != previous_pattern2:
                adjusted_s.iloc[i] = 0
                adjusted_s.iloc[i - 1] = 1

    return adjusted_s


def segmentgestalt(notematrix):
    """
    Segments the note matrix into meaningful groups based on IR patterns and boundaries.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes.

    Returns:
    list: A list of segmented DataFrames.
    """
    if notematrix.empty:
        return None

    notematrix = assign_ir_pattern_indices(notematrix)
    clind, clb = calculate_clang_boundaries(notematrix)
    s = calculate_segment_boundaries(notematrix, clind)
    s = adjust_segment_boundaries(notematrix, s)

    c = pd.Series(0, index=range(len(notematrix)))
    c.iloc[clind] = 1

    segments = []
    start_idx = 0
    for end_idx in s[s == 1].index:
        segments.append(notematrix.iloc[start_idx:end_idx + 1])
        start_idx = end_idx + 1
    segments.append(notematrix.iloc[start_idx:])

    return segments


def boundary(nmat, fig=False):
    """
    Local Boundary Detection Model by Cambouropoulos
    Returns the boundary strength profile of nmat according to the Local
    Boundary Detection Model by Cambouropoulos (1997)

    Parameters:
        nmat (pd.DataFrame): A DataFrame with columns for 'pitch', 'onset', and 'duration'.
        fig (bool): If True, creates a graphical output (default: False)

    Returns:
        np.ndarray: An array of boundary strengths (length equal to number of notes)

    Reference:
        Cambouropoulos, E. (1997). Musical rhythm: A formal model for determining local
        boundaries, accents and metre in a melodic surface. In M. Leman (Ed.),
        Music, Gestalt, and Computing: Studies in Cognitive and Systematic Musicology
        (pp. 277-293). Berlin: Springer Verlag.
    """
    # Extract pitch, onset, duration
    pitch = nmat['midi_pitch'].to_numpy()
    on = nmat['onset_beats'].to_numpy()
    dur = nmat['duration_beats'].to_numpy()
    off = on + dur

    # Profiles
    pp = np.abs(np.diff(pitch))  # pitch profile
    po = np.diff(on)  # IOI profile
    pr = np.maximum(0, on[1:] - off[:-1])  # rest profile

    # Degrees of change
    eps = 1e-6  # Small constant to prevent division by zero
    # Compute degrees of change and append zero to match lengths
    rp = np.concatenate((
        np.abs(pp[1:] - pp[:-1]) / (eps + pp[1:] + pp[:-1]),
        [0]
    ))
    ro = np.concatenate((
        np.abs(po[1:] - po[:-1]) / (eps + po[1:] + po[:-1]),
        [0]
    ))
    rr = np.concatenate((
        np.abs(pr[1:] - pr[:-1]) / (eps + pr[1:] + pr[:-1]),
        [0]
    ))

    # Strengths
    # Use concatenation to match MATLAB's indexing and dimensions
    sp = pp * np.concatenate((
        [0],
        rp[:-1] + rp[1:]
    ))
    if np.max(sp) > 0.1:
        sp = sp / np.max(sp)

    so = po * np.concatenate((
        [0],
        ro[:-1] + ro[1:]
    ))
    if np.max(so) > 0.1:
        so = so / np.max(so)

    sr = pr * np.concatenate((
        [0],
        rr[:-1] + rr[1:]
    ))
    if np.max(sr) > 0.1:
        sr = sr / np.max(sr)

    # Overall boundary strength profile
    b = np.concatenate((
        [1],
        0.25 * sp + 0.5 * so + 0.25 * sr
    ))

    if fig:
        # Create two subplots
        plt.figure(figsize=(12, 8))

        # Piano roll plot
        plt.subplot(2, 1, 1)
        for i in range(len(on)):
            plt.plot([on[i], on[i] + dur[i]], [pitch[i], pitch[i]], color='black')
        plt.title('Piano Roll')
        plt.ylabel('Pitch')
        xl = [np.min(on), np.max(on)]
        plt.xlim(xl)

        # Boundary strength plot
        plt.subplot(2, 1, 2)
        plt.stem(on, b, use_line_collection=True)
        plt.xlim(xl)
        plt.title('Boundary Strengths')
        plt.xlabel('Time')
        plt.ylabel('Strength')

        plt.tight_layout()
        plt.show()

    return b


def segment_lbdm(nmat):
    """
    Segments the note matrix based on boundary strengths calculated by the Local Boundary Detection Model,
    and adjusts the boundaries based on IR patterns.

    Parameters:
        nmat (pd.DataFrame): A DataFrame with columns 'pitch', 'onset', 'duration', and 'ir_symbol'.

    Returns:
        list of pd.DataFrame: A list where each element is a DataFrame representing a segment.
    """
    # Ensure 'ir_symbol' column exists
    if 'ir_symbol' not in nmat.columns:
        raise ValueError("The note matrix must contain an 'ir_symbol' column.")

    # Assign IR pattern indices
    nmat = assign_ir_pattern_indices(nmat)

    # Compute boundary strengths
    b = boundary(nmat)

    # Exclude the first element (always 1 in the model)
    b_no_first = b[1:]

    # Determine a threshold (e.g., 50% of the maximum boundary strength excluding the first element)
    max_b = np.max(b_no_first)
    threshold = 0.5 * max_b

    # Find indices where boundary strength exceeds the threshold
    boundary_indices = np.where(b_no_first > threshold)[0] + 1  # Adjust index since we excluded b[0]

    # Create initial segment boundary series
    s = pd.Series(0, index=range(len(nmat)))
    s.iloc[boundary_indices] = 1

    # Adjust segment boundaries based on IR patterns
    s = adjust_segment_boundaries(nmat, s)

    # Include start and end indices for segmentation
    segment_indices = [0] + s[s == 1].index.tolist() + [len(nmat)]

    # Remove duplicate indices and sort
    segment_indices = sorted(set(segment_indices))

    # Split the note matrix into segments
    segments = []
    for i in range(len(segment_indices) - 1):
        start_idx = segment_indices[i]
        end_idx = segment_indices[i + 1]
        segment = nmat.iloc[start_idx:end_idx].reset_index(drop=True)
        segments.append(segment)

    return segments


def preprocess_segments(segments: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Drops the pattern_index column and one-hot encodes the ir_symbol column for each DataFrame in the list of segments.

    Ensures that each DataFrame has columns for all specified states.

    Parameters:
    segments (list[pd.DataFrame]): List of DataFrames representing segments.

    Returns:
    list[pd.DataFrame]: List of preprocessed DataFrames.
    """
    # Define the possible states
    states = ['P', 'D', 'IP', 'ID', 'VP', 'R', 'IR', 'VR', 'M', 'd', 'rest']
    state_columns = [f'ir_symbol_{state}' for state in states]

    preprocessed_segments = []

    for segment in segments:
        # Drop the pattern_index column
        segment = segment.drop(columns=['pattern_index'])

        # One-hot encode the ir_symbol column
        segment = pd.get_dummies(segment, columns=['ir_symbol'])

        # Ensure all state columns are present
        for state_column in state_columns:
            if state_column not in segment.columns:
                segment[state_column] = 0
        segment[state_columns] = segment[state_columns].astype(int)

        # Reorder columns to ensure the state columns are in the correct order
        # 'onset_beats',
        # 'onset_beats_in_measure',
        # 'duration_beats',
        # 'midi_pitch',
        # 'pitch_class',
        # 'octave',
        # 'beat_strength'
        segment = segment[
            ['onset_beats_in_measure', 'duration_beats', 'pitch_class', 'octave', 'beat_strength',
             'expectancy'] + state_columns]

        preprocessed_segments.append(segment)

    return preprocessed_segments


def segments_to_distance_matrix(segments: list[pd.DataFrame], cores=None):
    """
    Converts segments to a distance matrix using multiprocessing.

    Parameters:
    segments (list[pd.DataFrame]): A list of segmented DataFrames.
    cores (int): The number of CPU cores to use for multiprocessing (default is None).

    Returns:
    np.ndarray: A distance matrix representing distances between segments.
    """
    if cores is not None and cores > cpu_count():
        raise ValueError(f"You don't have enough cores! Please specify a value within your system's number of "
                         f"cores. Core Count: {cpu_count()}")

    seg_np = [segment.to_numpy() for segment in segments]

    num_segments = len(seg_np)
    distance_matrix = np.zeros((num_segments, num_segments))

    args_list = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            args_list.append((i, j, segments[i], segments[j]))

    with Manager() as manager:
        message_list = manager.list()

        def log_message(message):
            message_list.append(message)

        with Pool(cores) as pool:
            results = pool.map(worker.calculate_distance, args_list)

        for i, j, distance, message in results:
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Reflect along the diagonal
            log_message(message)

        for message in message_list:
            print(message)

    return distance_matrix


def segments_to_graph(k: int, segments: list[pd.DataFrame], labeled_segments, cores=None):
    """
    Converts segments to a k-NN graph and ensures connectivity.

    Parameters:
    k (int): The number of neighbors for k-NN graph.
    segments (list[pd.DataFrame]): A list of segmented DataFrames.
    labeled_segments (list): A list of labeled segments.
    cores (int): The number of CPU cores to use for multiprocessing (default is None).

    Returns:
    tuple: A tuple containing:
        - networkx.Graph: The resulting k-NN graph.
        - np.ndarray: The distance matrix used to create the graph.
    """
    distance_matrix = segments_to_distance_matrix(segments, cores=cores)
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
    G = nx.from_scipy_sparse_array(knn_graph)

    for i in range(len(segments)):
        G.nodes[i]['segment'] = labeled_segments[i]

    if not nx.is_connected(G):
        print("The KNN graph is disjoint. Ensuring connectivity...")

        components = list(nx.connected_components(G))

        for i in range(len(components) - 1):
            min_dist = np.inf
            closest_pair = None
            for node1 in components[i]:
                for node2 in components[i + 1]:
                    dist = distance_matrix[node1, node2]
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (node1, node2)

            G.add_edge(closest_pair[0], closest_pair[1])

    return G, distance_matrix


def parse_segments(file_path):
    """
    Parses a text file with segments and numerical data into a numpy ndarray.

    Parameters:
    - file_path (str): Path to the input text file.

    Returns:
    - np.ndarray: 2D array where each row corresponds to a segment.
    """
    data = []  # List to hold all segments
    current_segment = []  # List to hold numbers of the current segment
    inside_brackets = False  # Flag to indicate if we are inside brackets

    # Regular expression to match 'Segment' lines
    segment_pattern = re.compile(r'^Segment\s+\d+')

    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, 1):
            stripped_line = line.strip()

            # Check if the line indicates a new segment
            if segment_pattern.match(stripped_line):
                if current_segment:
                    data.append(current_segment)
                    current_segment = []
                continue  # Move to the next line

            # Check if the line contains the start of data
            if '[' in stripped_line:
                inside_brackets = True
                # Remove everything before '['
                stripped_line = stripped_line.split('[', 1)[1]

            # Check if the line contains the end of data
            if ']' in stripped_line:
                # Remove everything after ']'
                stripped_line = stripped_line.split(']', 1)[0]
                inside_brackets = False

            if inside_brackets or stripped_line:
                # Split the line into individual numbers
                numbers_str = stripped_line.split()
                try:
                    # Convert string numbers to floats
                    numbers = [float(num) for num in numbers_str]
                    current_segment.extend(numbers)
                except ValueError as e:
                    print(f"Error parsing numbers on line {line_number}: {e}")
                    # Optionally, you can choose to exit or continue
                    continue

        # After reading all lines, append the last segment
        if current_segment:
            data.append(current_segment)

    # Convert the list of lists to a numpy ndarray
    array = np.array(data)

    return array


def mobility(nmat: pd.DataFrame):
    """
    Melodic motion as a mobility (Hippel, 2000).
    Based on mobility function from MidiToolKit (Toiviainen. 2016)
    """
    if len(nmat) == 0:
        return np.array([])

    pitches = nmat['midi_pitch'].to_numpy()
    n = len(nmat.index)

    if n < 2:
        return np.zeros(n)

    mob = np.zeros(n)
    y = np.zeros(n)

    for i in range(1, n):
        mean_pitch = np.mean(pitches[:i])

        p = pitches[:i] - mean_pitch
        p2 = pitches[1:i + 1] - mean_pitch

        if len(p) > 1:
            correlation_matrix = np.corrcoef(p, p2)
            mob[i] = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0

        y[i - 1] = mob[i - 1] * (pitches[i] - mean_pitch)

    y[-1] = 0

    return np.abs(y)


def tessitura(nmat: pd.DataFrame):
    """
    Melodic tessitura based on deviation from median pitch height (Hippel, 2000)
    Based on tessitura function from MidiToolKit (Toiviainen. 2016)
    """
    if len(nmat) == 0:
        return np.array([])

    n = len(nmat.index)
    if n < 2:
        return np.zeros(n)
    pitches = nmat['midi_pitch'].to_numpy()

    deviation = np.zeros(n)
    y = np.zeros(n)

    for i in range(1, n):
        median_pitch = np.median(pitches[:i])
        deviation[i - 1] = np.std(pitches[:i])
        if deviation[i - 1] == 0:
            y[i - 1] = 0  # If no variation, set tessitura to 0
        else:
            y[i - 1] = (pitches[i] - median_pitch) / deviation[i - 1]
        # y[i-1] = (pitches[i] - median_pitch) / deviation[i-1]
        y[0] = 0

    return np.abs(y)


def calculate_note_expectancy_scores(nmat: pd.DataFrame) -> np.ndarray:
    w1 = 0.7
    w2 = 0.3

    # nmat['tessitura'].replace([np.inf, -np.inf], np.nan, inplace=True)
    # nmat['mobility'].replace([np.inf, -np.inf], np.nan, inplace=True)
    # nmat.fillna(nmat.mean(), inplace=True)

    scaler = MinMaxScaler()
    normalized_tessitura = scaler.fit_transform(nmat[['tessitura']])
    normalized_mobility = scaler.fit_transform(nmat[['mobility']])
    raw_expectancy = w1 * (1 - normalized_tessitura) + w2 * normalized_mobility
    expectancy_scores = 0.5 + 0.5 * (raw_expectancy - 0.5)

    # Adjust first two notes
    expectancy_scores[0] = 0.5

    return expectancy_scores
