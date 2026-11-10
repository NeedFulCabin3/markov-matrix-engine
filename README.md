# Markov Matrix Engine

A pragmatic, production-ready discrete-time Markov Chain modeling system engineered for numerical state forecasting and long-term steady-state analysis using NumPy.

## Overview

In operational forecasting, financial modeling, and queueing theory, tracking discrete state transitions over time often suffers from manual matrix multiplication errors, floating-point drift, and silent stochastic constraint violations. This tool solves these operational pain points by providing a strict, validated mathematical engine that enforces stochastic properties at runtime, ensuring that probability distributions remain mathematically sound across multi-step projections.

## How It Works

The engine operates on discrete-time Markov chains utilizing row vector conventions ($v_n = v_0 P^n$). 
1. **Stochastic Validation:** During initialization, the input transition matrix is checked for square dimensions, boundary constraints ($0 \le P_{ij} \le 1$), and row-sum unity using `math.isclose()` with tight floating-point tolerances.
2. **Temporal Prediction:** Future state probability distributions are computed efficiently by raising the transition matrix to the power of $n$ using NumPy's optimized linear algebra routines (`np.linalg.matrix_power`), followed by a dot product with the initial state vector.
3. **Steady-State Convergence:** Long-term equilibrium is calculated through iterative vector-matrix multiplications, halting when successive iterations fall below a defined absolute tolerance (`np.allclose`), or raising a custom `ConvergenceError` if the system fails to stabilize.

## Key Features

* **Strict Stochastic Verification:** Automatically catches invalid transition probabilities or rows that fail to sum to 1.0.
* **Vectorised Projections:** Computes multi-step state probabilities in logarithmic time complexity relative to the number of steps using matrix exponentiation.
* **Iterative Equilibrium Solver:** Determines long-term steady-state distributions with configurable maximum iteration bounds and tolerance thresholds.
* **Custom Exception Handling:** Distinguishes between configuration errors (`ValueError`) and numerical non-convergence (`ConvergenceError`).

## Tech Stack & Core Dependencies Breakdown

* **Python 3.10+:** Utilised for modern type hint syntax (`list[float] | np.ndarray`).
* **NumPy (`numpy`):** Chosen for high-performance, C-backed array slicing, matrix exponentiation, and dot-product calculations.
* **Standard Library (`math`):** Leveraged for robust floating-point equality comparisons (`math.isclose`).

## Environment & Web-Based Quick Start

You can run and test this repository directly inside your browser using GitHub Codespaces without installing local terminal tools.

### 1. Open in GitHub Codespaces
Click the **Code** dropdown on the repository main page, select the **Codespaces** tab, and click **Create codespace on main**.

### 2. Set Up Virtual Environment (Optional / Local)
If working inside a terminal environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy
```

### 3. Execution
Run the embedded weather prediction example script:
```bash
python main.py
```

## Repository Structure

```bash
markov-matrix-engine/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── main.py
└── README.md
```

## Roadmap

**[ ] Type Stub Integration (.pyi):** Introduce standalone type stub files to improve IDE type inference for downstream consumers.

**[ ] Sparse Matrix Support:** Migrate transition matrix operations to scipy.sparse for handling massive state spaces with high zero-probability densities.

**[ ] Structural Logging:** Replace standard print statements with structured logging handlers for audit trails in production pipelines.