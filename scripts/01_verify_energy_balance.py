import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import collapse_solver

def run_energy_check():
    print("Running Energy Conservation Check (Stable Regime)...")
    
    # Amplitud MUY baja (0.02) -> Energía pequeña (~0.1) -> Estable
    # IMPORTANTE: Asegúrate de que 'Simulation' existe en collapse_solver
    sim = collapse_solver.Simulation(gamma=0.05, N=400, r_max=20.0, pulse_amp=0.02)
    
    initial_energy = sim.calculate_total_energy()
    print(f"Initial Energy: {initial_energy:.5f}") 
    
    times, E_geo, E_diss = [], [], []
    
    # Correr simulación
    for i in range(500):
        sim.step()
        if i % 10 == 0:
            times.append(sim.t)
            E_geo.append(sim.calculate_total_energy())
            E_diss.append(sim.dissipated_energy_accumulated)
    
    E_geo = np.array(E_geo)
    E_diss = np.array(E_diss)
    E_total = E_geo + E_diss
    
    plt.figure()
    plt.plot(times, E_geo/initial_energy, label='Geometric Energy')
    plt.plot(times, E_diss/initial_energy, label='Dissipated Energy')
    plt.plot(times, E_total/initial_energy, 'k--', label='Total (Sum)')
    plt.legend()
    plt.title('Covariance Check: Energy Balance')
    plt.ylabel('Normalized Energy')
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'fig_energy_check.pdf')
    plt.savefig(out_path)
    
    err = abs(E_total[-1] - initial_energy)
    print(f"Final Absolute Error: {err:.5f}")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    run_energy_check()