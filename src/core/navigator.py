from typing import List, Dict, Tuple
import numpy as np
from .gradient import OziresAGradient


def generate_future_paths(tensor: dict, collapse_points: List[str]) -> List[Dict]:
    """
    Defines symbolic future scenarios based on current collapse map.
    """
    paths = []

    base = {
        'name': "Base Scenario",
        'growth': 1.02,
        'stability': 0.85,
        'ethical_risk': 0.30,
        'avoided_collapses': [p for p in collapse_points if tensor[p][4] > 0.8],
        'key_factors': ["IBTX-42", "YUANX", "IES-NY"]
    }

    collapse = {
        'name': "Collapse Scenario",
        'growth': 0.75,
        'stability': 0.40,
        'ethical_risk': 0.90,
        'avoided_collapses': [],
        'key_factors': ["BraneCoin", "IES-SP", "IES-Nairobi"]
    }

    disruptive = {
        'name': "Disruptive Scenario",
        'growth': 1.25,
        'stability': 0.55,
        'ethical_risk': 0.60,
        'avoided_collapses': ["BraneCoin", "IES-Nairobi"],
        'key_factors': ["Solitarium", "Clean Water", "Synthetic Lithium"]
    }

    return [base, collapse, disruptive]


def evaluate_path_score(path: Dict, current_state: float, gradient: OziresAGradient) -> float:
    """
    Applies the symbolic decision metric with ethical-temporal modulation.
    """
    collapse_magnitude = current_state / 10.0  # normalize for tau=1
    force = gradient.compute(collapse_magnitude)
    force_weight = np.dot(force, [1, 1, 1])  # simple sum of contributions

    return (
        path['growth'] * 0.5 +
        path['stability'] * 0.3 -
        path['ethical_risk'] * 0.2 +
        force_weight * 0.1
    )


def select_optimal_path(paths: List[Dict], current_state: float) -> Tuple[Dict, float]:
    """
    Returns the path with the highest symbolic value, adjusted by gradient.
    """
    gradient = OziresAGradient()
    scores = [evaluate_path_score(p, current_state, gradient) for p in paths]
    best_index = int(np.argmax(scores))
    return paths[best_index], scores[best_index]
