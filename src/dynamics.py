import numpy as np
from scipy.integrate import solve_ivp

# PARÁMETROS GLOBALES ÚNICOS
DEFAULT = {
    'rtol': 1e-8,
    'atol': 1e-10,
    'T_max': 30.0,
    'escape_delta': 0.02,
    'pulse_t0': 5.0,
    'pulse_width': 2.0
}

def flow(t, y, gamma, V_prime_func, F=0.0, pulse=False):
    """
    Sistema dinámico: φ̈ + γφ̇ + V'(φ) = F(t)
    y = [φ, φ̇]
    """
    phi, v = y
    
    # Término de pulso (para energy scan)
    if pulse and F != 0:
        pulse_term = F * np.exp(-DEFAULT['pulse_width'] * (t - DEFAULT['pulse_t0'])**2)
    else:
        pulse_term = 0.0
    
    dphi = v
    dv = pulse_term - V_prime_func(phi) - gamma * v
    
    return [dphi, dv]

def simulate_escape(phi0, v0, gamma, V_prime_func, saddle_pos, **kwargs):
    """Simula y verifica si escapa cruzando el saddle"""
    delta = kwargs.get('escape_delta', DEFAULT['escape_delta'])
    T = kwargs.get('T_max', DEFAULT['T_max'])
    
    def escape_event(t, y):
        return y[0] - (saddle_pos + delta)
    escape_event.terminal = True
    escape_event.direction = 1.0
    
    sol = solve_ivp(
        lambda t, y: flow(t, y, gamma, V_prime_func),
        [0, T],
        [phi0, v0],
        events=escape_event,
        method='Radau',
        rtol=DEFAULT['rtol'],
        atol=DEFAULT['atol']
    )
    
    return len(sol.t_events[0]) > 0, sol