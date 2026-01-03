"""
SMQTK Metrics - Distance and similarity functions.
"""
from math import acos, pi
import numpy as np


def histogram_intersection_distance(a, b):
    """
    Compute the histogram intersection distance between given histogram
    vectors or matrices a and b, returning a value between 0.0 and 1.0.
    0.0 means full intersection, 1.0 means no intersection.
    """
    sum_axis = 1
    if a.ndim == 1 and b.ndim == 1:
        sum_axis = 0
    return 1. - ((np.add(a, b) - np.abs(np.subtract(a, b))).sum(sum_axis) * 0.5)


def euclidean_distance(i, j):
    """
    Compute euclidean distance between two N-dimensional point vectors.
    """
    sum_axis = 1
    if i.ndim == 1 and j.ndim == 1:
        sum_axis = 0
    return np.sqrt(np.square(i - j).sum(sum_axis))


def cosine_similarity(i, j):
    """
    Angular similarity between vectors i and j. Results in a value between 1
    (exactly the same) to -1 (exactly opposite). 0 indicates orthogonality.
    """
    return np.dot(i, j) / (np.sqrt(i.dot(i)) * np.sqrt(j.dot(j)))


def cosine_distance(i, j, pos_vectors=True):
    """
    Cosine similarity converted into angular distance.
    """
    sim = max(-1.0, min(cosine_similarity(i, j), 1))
    return (1 + bool(pos_vectors)) * acos(sim) / pi


def hamming_distance(i, j):
    """
    Return the hamming distance between the two given integers,
    or the number of places where the bits differ.
    """
    return bin(i ^ j).count('1')


__all__ = [
    'histogram_intersection_distance',
    'euclidean_distance',
    'cosine_similarity',
    'cosine_distance',
    'hamming_distance',
]
