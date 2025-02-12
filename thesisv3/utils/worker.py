from tslearn.metrics import dtw


def calculate_distance(args):
    i, j, segment_i, segment_j = args
    distance = dtw(segment_i, segment_j)
    message = f"Calculated Edge Weight ({i}, {j})"
    return i, j, distance, message
