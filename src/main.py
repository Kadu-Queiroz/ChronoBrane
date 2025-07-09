"""
ChronoBrane Main Entry Point
----------------------------
This script loads a spacetime tensor (real or simulated), detects collapse points
(entropic spikes), and navigates potential futures using the ChronoBrane system.

Modules:
- Detector: identifies high-curvature regions (potential crises).
- Gradient: ethical vector modulation (Ozires-A gradient).
- Navigator: generates and selects optimal paths avoiding collapses.

Run this as a conceptual simulation and demonstration of the ChronoBrane pipeline.
"""

from src.core.gradient import OziresAGradient
from src.core.navigator import generate_future_paths, select_optimal_path
from src.simulation.crisis_2008_sim import load_tensor_data, detect_collapses


def main():
    # Load a synthetic or real spacetime tensor (e.g., economic crisis of 2008)
    tensor = load_tensor_data()

    # Detect collapse points using curvature and entropy
    collapse_points, curvature, entropy = detect_collapses(tensor)

    # Define current conceptual state (mean entropy or symbolic metric)
    current_state = float(sum(entropy)) / len(entropy)

    # Initialize Ozires-A ethical gradient
    gradient = OziresAGradient()
    navigation_vector = gradient.compute(collapse_distance=0.3)
    print(f"Ethical Navigation Vector (∇𝒫ₒₓᵢᵣₑₛ₋ₐ): {navigation_vector}")

    # Generate symbolic future paths
    paths = generate_future_paths(tensor, collapse_points)

    # Select the optimal path based on current state
    best_path, score = select_optimal_path(paths, current_state)

    # Report outcome
    print("\n=== CHRONOS-BRANE DECISION REPORT ===")
    print(f"Collapse Points Detected: {collapse_points}")
    print(f"Optimal Path: {best_path['name']} (Score: {score:.2f})")
    print(f"Key Factors: {best_path['key_factors']}")
    print(f"Avoided Collapses: {best_path['avoided_collapses']}\n")


if __name__ == "__main__":
    main()