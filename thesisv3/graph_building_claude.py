"""
Graph Building Module for Music Analysis

This module provides functions for building and analyzing graphs from musical segments.
It includes utilities for distance matrix calculation, graph construction, spectral analysis,
and comparison of musical styles between artists.
"""

import os

from networkx import Graph
from numpy.linalg import eigh
from scipy.sparse import csgraph
from sklearn.manifold import MDS

from preprocessing.preprocessing import *
from music21 import converter, environment
from grakel.kernels import WeisfeilerLehman
import music21
import pickle

# Configure MuseScore paths
env = environment.Environment()
env['musicxmlPath'] = 'C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe'
env['musescoreDirectPNGPath'] = 'C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe'

us = music21.environment.UserSettings()
us['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'


# ===============================
# File I/O Operations
# ===============================




