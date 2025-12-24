import sys
import os
import numpy as np
import matplotlib.pyplot as plt

SRC_PATH = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(SRC_PATH)

import collapse_solver

# --- Resolver nombre de la clase de simulación ---
if hasattr(collapse_solver, "Simulation"):
    Simulation = collapse_solver.Simulation
elif hasattr(collapse_solver, "CollapseSimulation"):
    Simulation = collapse_solver.CollapseSimulation
else:
    raise ImportError(
        "No Simulation class found in collapse_solver.py. "
        "Available symbols: "
        + ", ".join(dir(collapse_solver))
    )

def effective_kretschmann(sim):
    Pi = sim.Pi
    dphi = np.gradient(sim.phi, sim.dr)
    V = 0.5 * sim.phi**2
    rho = Pi**2 + dphi**2 + V
    return np.max(rho**2)

def get_curvature_trace(gamma_val):
    print(f"Running simulation for gamma = {gamma_val}")
    
    sim = Simulation(gamma=gamma_val, N=200, r_max=20.0, pulse_amp=0.4)
    
    t, K = [], []

    for _ in range(2500):
        sim.step()
        Kmax = effective_kretschmann(sim)
        t.append(sim.t)
        K.append(Kmax)
        if Kmax > 1e6:
            break

    return np.array(t), np.array(K)

def plot_curvature_evolution():
    t0, K0 = get_curvature_trace(0.0)
    t1, K1 = get_curvature_trace(0.5)

    plt.figure(figsize=(7,5))
    plt.plot(t0, K0, 'r--', lw=2, label=r'$\gamma=0$')
    plt.plot(t1, K1, 'b-',  lw=2, label=r'$\gamma>\gamma_c$')
    plt.yscale('log')
    plt.xlabel(r'$t/M$')
    plt.ylabel(r'$K_{\mathrm{eff}}$')
    plt.legend()
    plt.grid(alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), '..', 'figures', 'fig_curvature_dynamics.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    print(f"Saved {out}")

if __name__ == "__main__":
    plot_curvature_evolution()
