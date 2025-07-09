import numpy as np

class OziresAGradient:
    """Implements the ethical-temporal navigation vector: ∇𝒫ₒₓᵢᵣₑₛ₋ₐ"""

    def __init__(self, vector_components=None, tau=1.0):
        """
        vector_components: list or array [growth, stability, ethics]
        tau: normalizing factor (max time to collapse)
        """
        self.V = np.array(vector_components if vector_components else [0.25, -0.40, 0.18])
        self.tau = tau

    def compute(self, collapse_distance):
        """
        Computes the ethical navigation vector given a collapse distance.

        collapse_distance: float (0 to tau)
        returns: np.array of ∇𝒫 (navigation force vector)
        """
        factor = 1 - (collapse_distance / self.tau)
        return self.V * factor

    def magnitude(self, collapse_distance):
        return np.linalg.norm(self.compute(collapse_distance))
