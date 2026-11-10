import math
import numpy as np

class ConvergenceError(Exception):
    """Raised when iterative numerical algorithms fail to converge."""

    pass


class MarkovChainPredictor:
    """A pragmatic discrete-time Markov Chain modeling system."""

    def __init__(self, transition_matrix: list[list[float]] | np.ndarray):
        self.P = np.array(transition_matrix, dtype=float)
        self._validate_stochastic_matrix()

    def _validate_stochastic_matrix(self) -> None:
        """Validates that the input is a square matrix representing a valid stochastic matrix."""
        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError("Transition matrix must be a square 2D array.")

        n = self.P.shape[0]
        for i in range(n):
            row = self.P[i]
            if np.any(row < 0.0) or np.any(row > 1.0):
                raise ValueError(
                    f"Row {i} contains probability values outside the [0, 1] range."
                )
            
            row_sum = float(np.sum(row))
            if not math.isclose(row_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(
                    f"Row {i} sum ({row_sum:.5f}) does not equal 1.0. Stochastic property violated."
                )

    def predict(self, initial_state: list[float] | np.ndarray, steps: int) -> np.ndarray:
        """Computes the state probability distribution after 'steps' discrete intervals.
        
        Uses row vector convention: v_n = v_0 * (P ^ n)
        """
        if steps < 0:
            raise ValueError("Time steps cannot be negative.")

        v0 = np.array(initial_state, dtype=float)
        if v0.ndim != 1 or v0.shape[0] != self.P.shape[0]:
            raise ValueError("Initial state vector dimension must match matrix size.")

        v0_sum = float(np.sum(v0))
        if not math.isclose(v0_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5) or np.any(v0 < 0.0):
            raise ValueError("Initial state vector must be a valid probability distribution (sums to 1).")

        # Compute P^steps efficiently using NumPy matrix power and compute dot product
        p_n = np.linalg.matrix_power(self.P, steps)
        return np.dot(v0, p_n)

    def steady_state(self, max_iter: int = 1000, tol: float = 1e-7) -> np.ndarray:
        """Calculates the long-term steady-state distribution by iterating matrix powers / vector updates."""
        n = self.P.shape[0]
        v = np.ones(n) / n  # Initialize with a uniform probability distribution

        for _ in range(max_iter):
            v_next = np.dot(v, self.P)
            if np.allclose(v, v_next, atol=tol):
                return v_next
            v = v_next

        raise ConvergenceError("Steady-state distribution failed to converge within maximum iterations.")


if __name__ == "__main__":
    # Example: Weather Prediction Model
    # States: 0 = Sunny, 1 = Rainy
    # Transition Rules:
    # - From Sunny: 80% chance stay Sunny, 20% chance turn Rainy
    # - From Rainy: 40% chance turn Sunny, 60% chance stay Rainy
    transition_matrix = [
        [0.8, 0.2],
        [0.4, 0.6]
    ]

    try:
        model = MarkovChainPredictor(transition_matrix)

        # Initial state: Today is 100% Sunny [Sunny, Rainy]
        initial_state = [1.0, 0.0]
        steps = 3

        future_distribution = model.predict(initial_state, steps)
        steady_state_dist = model.steady_state()

        print("--- Markov Chain Predictor Results ---")
        print(f"Initial State: {initial_state}")
        print(f"Distribution after {steps} steps: {np.round(future_distribution, 4)}")
        print(f"Steady-State Distribution:    {np.round(steady_state_dist, 4)}")

    except (ValueError, ConvergenceError) as err:
        print(f"Modeling Error: {err}")