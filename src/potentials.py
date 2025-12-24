import numpy as np

def V_double_well(phi, asymmetry=0.8):
    """Potencial asimétrico: φ⁴ - 2φ² + aφ"""
    return phi**4 - 2*phi**2 + asymmetry*phi

def V_prime_double_well(phi, asymmetry=0.8):
    """Derivada: 4φ³ - 4φ + a"""
    return 4*phi**3 - 4*phi + asymmetry

def V_double_prime_double_well(phi):
    """Segunda derivada: 12φ² - 4"""
    return 12*phi**2 - 4

# Para universalidad: otros potenciales
def V_prime_symmetric(phi):
    """Doble pozo simétrico"""
    return 4*phi**3 - 4*phi

def V_prime_exponential(phi, V0=0.5, lambd=1.0):
    """Potencial exponencial confinante"""
    return V0 * lambd * (np.exp(lambd*phi) - 1)

# Diccionario para fácil acceso
POTENTIALS = {
    "asymmetric": V_prime_double_well,
    "symmetric": V_prime_symmetric,
    "exponential": V_prime_exponential
}