
import numpy as np
from scipy import signal


def detect_collapses(tensor: dict, threshold: float = 0.5):
    """
    Identifies collapse points based on the curvature of entropy.

    Args:
        tensor: Dictionary with entries [region, type, time, magnitude, entropy]
        threshold: Threshold to consider a point as a collapse (based on curvature)

    Returns:
        Tuple containing:
            - List of keys (names) of the points classified as collapses
            - Curvature vector
            - Original entropy values
    """
    entropies = np.array([v[4] for v in tensor.values()])
    names = list(tensor.keys())

    first_derivative = np.gradient(entropies)
    curvature = np.gradient(first_derivative)

    peaks, _ = signal.find_peaks(np.abs(curvature), height=threshold)
    collapse_points = [names[i] for i in peaks]

    return collapse_points, curvature, entropies
