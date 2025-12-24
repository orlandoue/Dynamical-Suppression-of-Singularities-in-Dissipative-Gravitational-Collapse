# Dynamical Suppression of Singularities in Dissipative Gravitational Collapse

**Orlando Miguel Urbina Gonzalez** *Facultad de Física, Pontificia Universidad Católica de Chile* [Physical Review D - Submitted]

## Abstract
This repository contains the numerical implementation accompanying the paper **"Dynamical Suppression of Singularities in Dissipative Gravitational Collapse"**.

We demonstrate a dynamical mechanism that prevents the formation of spacetime singularities using an **Effective Field Theory (EFT)** approach. By modeling the horizon interaction as a covariant dissipative fluid (inspired by the Membrane Paradigm) and incorporating the **Tolman gravitational blue-shift**, we show that the dissipative backreaction diverges in the strong-field regime, halting the collapse and regularizing the geometry.

## Key Results reproduced by this code:
1.  **Dynamical Bifurcation:** Existence of a critical threshold $\gamma_c$ separating singular collapse from regular thermalization.
2.  **Basin Contraction:** Stochastic analysis showing how the phase-space volume of singular trajectories contracts to zero.
3.  **Covariance Check:** Numerical verification of energy conservation (Bianchi identities) accounting for the heat flow.

## Repository Structure

- `src/`: Core physics engines.
    - `collapse_solver.py`: 1D Spherically symmetric Einstein-Klein-Gordon solver with ADM formalism.
    - `dynamics.py` & `potentials.py`: Stochastic Langevin dynamics for phase-space basin analysis.
- `scripts/`: Generation scripts for each figure in the manuscript.
- `paper/`: PDF of the manuscript.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/tu-usuario/dissipative-collapse-prd.git](https://github.com/tu-usuario/dissipative-collapse-prd.git)
   cd dissipative-collapse-prd
