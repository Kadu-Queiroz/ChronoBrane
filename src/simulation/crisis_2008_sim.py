import numpy as np
from scipy import signal

def load_tensor_data():
    """
    Loads synthetic or historical tensor data.
    Returns a dictionary with entries in the format:
    { label: [region, type, time, magnitude, entropy] }
    """
    return {
        "IBTX-42":        [1.0, 0.1, 10.05, 1.37, 0.12],
        "NASDAQ Quantum": [2.0, 0.1, 10.05, -0.65, 0.85],
        "YUANX":          [3.0, 0.1, 10.05, 2.12, 0.08],
        "ChronosToken":   [1.0, 0.2, 10.05, 6.80, 0.15],
        "BraneCoin":      [2.0, 0.2, 10.05, -3.20, 0.92],
        "Solitarium":     [3.0, 0.2, 10.05, 14.30, 0.25],
        "Lítio Sintético": [1.0, 0.3, 10.05, 9.00, 0.18],
        "Água Potável":   [2.0, 0.3, 10.05, 1.10, 0.75],
        "IES-SP":         [1.0, 0.4, 10.05, 7.80, 0.95],
        "Trend-Valvet":   [2.0, 0.4, 10.05, 210.0, 0.88],

        # Real historical crisis indicators for testing (2008 financial crisis)
        "SP500-2006":     [1.0, 0.1, 2006.0, 1.42, 0.25],
        "SP500-2007":     [1.0, 0.1, 2007.0, 1.20, 0.55],
        "SP500-2008":     [1.0, 0.1, 2008.0, 0.65, 0.92],
        "HousingIndex-2006": [1.0, 0.3, 2006.0, 2.50, 0.15],
        "HousingIndex-2007": [1.0, 0.3, 2007.0, 2.10, 0.35],
        "HousingIndex-2008": [1.0, 0.3, 2008.0, 1.05, 0.85]
    }

def detect_collapses(tensor: dict, threshold: float = 0.5):
    """
    Identifies high-curvature points based on entropy variation.
    Returns list of collapse labels, full curvature array, and entropy array.
    """
    entropy = np.array([v[4] for v in tensor.values()])
    labels = list(tensor.keys())

    first_derivative = np.gradient(entropy)
    curvature = np.gradient(first_derivative)

    peaks, _ = signal.find_peaks(np.abs(curvature), height=threshold)
    collapse_labels = [labels[i] for i in peaks]

    return collapse_labels, curvature, entropy
