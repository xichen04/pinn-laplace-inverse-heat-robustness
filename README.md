# Robustness of PINNs for the 2D Laplace Equation

This repository contains the code, report, and experiment outputs for a MATH3349 project evaluating how well Physics-Informed Neural Networks (PINNs) reconstruct a steady-state 2D temperature field governed by Laplace's equation when only sparse and noisy measurements are available.

The central question is practical: if a full thermal boundary cannot be measured, can a PINN recover the whole temperature field from a small number of interior sensors, partial boundary data, and the physics constraint?

## Experiment Overview

The physical system is a square domain:

```text
Omega = [0, 1] x [0, 1]
```

with the steady-state Laplace equation:

```text
d2u/dx2 + d2u/dy2 = 0
```

A finite-difference method (FDM) solution is used as the reference field. PINNs are then trained as inverse solvers using combinations of:

- Sensor density: 5x5 and 10x10 interior sensor grids
- Boundary availability: partial boundary conditions versus no boundary conditions
- Sensor noise: 0%, 1%, 2%, 5%, and 10% Gaussian noise
- Repeated trials: 10 independent runs per setting

The PINN is an 8-hidden-layer multilayer perceptron with 20 neurons per layer and `tanh` activations. It is trained with Adam at a learning rate of `1e-3` for 50,000 epochs. Accuracy is measured using relative L2 error against the FDM reference solution on a 100x100 grid.

## Key Findings

The experiments show that PINNs can reconstruct the temperature field accurately even when the available measurements are sparse or noisy. The main trends are:

- Increasing sensor density from 5x5 to 10x10 generally improves both accuracy and stability.
- Increasing Gaussian noise raises the relative L2 error, as expected.
- Partial boundary conditions improve robustness in harder, noisier settings.
- In very low-noise cases, the no-boundary setup can sometimes perform better because it gives the model more flexibility and avoids an additional boundary-loss constraint.
- The problem remains learnable because the true Laplace solution is smooth and low-complexity, so the PDE residual plus sparse sensor data provides a strong regularizing signal.

These results suggest that PINNs are useful for thermal inverse problems where full boundary measurements are expensive, unavailable, or contaminated by noise.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── reports/
│   └── math3349_project_report.pdf
├── src/
│   └── math3349_code.py
└── results/
    ├── 5x5/
    ├── 10x10/
    └── comparisons/
```

## Selected Results

The full report is available at [`reports/math3349_project_report.pdf`](reports/math3349_project_report.pdf).

Important visual outputs are included in:

- [`results/5x5/`](results/5x5/) for sparse-sensor experiments
- [`results/10x10/`](results/10x10/) for denser-sensor experiments
- [`results/comparisons/`](results/comparisons/) for summary comparisons

The summary table reports mean and standard deviation of relative L2 error across grid size, boundary-condition type, and noise level.

## Running the Code

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python src/math3349_code.py
```

The full experiment is computationally expensive because it trains many PINNs for 50,000 epochs each. The original experiments were run on Google Colab CPU, with each training run taking approximately 35-40 minutes.

## Notes

The script was exported from a Colab notebook and includes the full workflow:

1. Construct an FDM reference solution for the 2D Laplace equation.
2. Define the PINN architecture and Laplace residual using JAX automatic differentiation.
3. Generate sparse sensor observations and optional partial boundary constraints.
4. Add Gaussian noise to sensor measurements.
5. Train repeated PINN trials and compare predictions to the FDM reference.
6. Plot prediction fields, absolute error maps, boxplots, and summary tables.

## Reference

This project follows the Physics-Informed Neural Network framework introduced by Raissi, Perdikaris, and Karniadakis:

Raissi, M., Perdikaris, P., and Karniadakis, G. E. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics, 378, 686-707, 2019.
