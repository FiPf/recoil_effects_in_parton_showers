__author__ = 'Fiona C. Pärli'
__email__ = 'fiona.paerli@students.unibe.ch'
__date__ = 'June 2025'

import numpy as np
from typing import Callable

def monte_carlo_integrator(func: Callable, a: float, b: float, num_samples: int) -> tuple[float, float]: 
    """general Monte Carlo integration for any function

    Args:
        func (Callable): function that should be integrated
        a (float): lower integration boundary
        b (float): upper integration boundary
        num_samples (int): number of Monte Carlo samples

    Returns:
        (integral, error (tuple[float, float]): integral and its Monte Carlo error 
    """
    x_samples = np.random.uniform(a, b, num_samples)
    y_samples = np.array([func(x) for x in x_samples]) 
    integral = (b - a) * np.mean(y_samples)
    error = (b - a) * np.std(y_samples) / np.sqrt(num_samples)  # Monte Carlo error formula
    return integral, error

def test_func(x):
    return x**2

def four_dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Four vector dot product with the Minkowski metric

    Args:
        vec1 (np.ndarray): First four vector
        vec2 (np.ndarray): Second four vector

    Returns:
        (float): result of the Minkowski dot product
    """
    metric = np.array([1, -1, -1, -1])
    return np.sum(vec1 * vec2 * metric)

def angle_integral(n1: np.ndarray, n2: np.ndarray, alpha: float, num_samples: int=100_000):
    """Uses Monte Carlo method to calculate the angle integral

    Args:
        n1 (np.ndarray): first event vector after the transformations
        n2 (np.ndarray): second event vector after the transformations
        alpha (float): opening angle of the jets
        num_samples (int, optional): Number of Monte Carlo samples. Defaults to 100_000.

    Returns: 
        (float): integral
        (float): Monte Carlo error
    """
    def integrand(_):
        theta = np.arccos(np.cos(alpha) - (np.cos(alpha) - np.cos(np.pi - alpha)) * np.random.rand()) #no need to multiply the integrand by sin(theta) because of this
        phi = 2 * np.pi * np.random.rand()

        W = calculate_W(n1, n2, theta, phi)

        return W

    integral, error = monte_carlo_integrator(integrand, 0, 1, num_samples)

    theta_range = np.cos(alpha) - np.cos(np.pi - alpha)
    prefactor = theta_range

    return prefactor * integral, prefactor * error

def transformed_angle_integral(n1: np.ndarray, n2: np.ndarray, alpha: float, num_samples=100_000):
    """Uses Monte Carlo method to calculate the angle integral after the variable transformation

    Args:
        n1 (np.ndarray): first event vector after the transformations
        n2 (np.ndarray): second event vector after the transformations
        alpha (float): opening angle of the jets
        num_samples (int, optional): Number of Monte Carlo samples. Defaults to 100_000.

    Returns: 
        (float): integral
        (float): Monte Carlo error
    """
    y_min = 0.5 * np.log((1 + np.cos(alpha)) / (1 - np.cos(alpha)))
    y_max = 0.5 * np.log((1 + np.cos(np.pi - alpha)) / (1 - np.cos(np.pi - alpha)))
    
    def integrand(y):
        theta = np.arccos((np.exp(2 * y) - 1) / (np.exp(2 * y) + 1))
        phi = 2 * np.pi * np.random.rand()

        W = calculate_W(n1, n2, theta, phi)

        jacobian = 2*np.sqrt((np.exp(2*y)/(np.exp(2*y) + 1)**2))

        return W*jacobian*np.sin(theta)
    
    integral, error = monte_carlo_integrator(integrand, 0, 1, num_samples)
    prefactor = abs((y_max - y_min))

    return prefactor * integral, prefactor * error

def calculate_W(n1: np.ndarray, n2: np.ndarray, theta: float, phi: float): 
    """calculate W_ij 

    Args:
        n1 (np.ndarray): first event vector after the transformations
        n2 (np.ndarray): second event vector after the transformations
        theta (float): angle 
        phi (float): angle

    Returns:
        W (float): calculated W
    """
    n1_dot_n2 = four_dot_product(n1, n2)

    nq = np.array([1, np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    n1_dot_nq = four_dot_product(n1, nq)
    n2_dot_nq = four_dot_product(n2, nq)

    W = (n1_dot_n2)/(n1_dot_nq*n2_dot_nq)
    return W

import numpy as np
import math

def analytical_res(alpha: float):
    """Analytical result of the first test angular integral. Derivation see separate document. 

    Args:
        alpha (float): opening angle of the jet

    Returns:
        res (float): result of the integration
    """
    up = np.pi - alpha
    low = alpha

    def f(x): 
        return 2*np.log(np.tan(x/2))
    
    res = f(up) - f(low)

    return res

#analytical result of toy model for any n and sum of all n solutions
def toy_model_sol(n: int, t: float, V: float, R: float): 
    if n == 2:
        return np.exp(-2 * V * t)
    else:
        factor = (R / V) ** (n - 2)
        return factor * np.exp(-n * V * t) * (1 - np.exp(V * t)) ** (n - 2)/math.factorial(n-2)

def all_order_sum(t: float, V: float, R: float):
    """Closed form S(t) = sum_{n>=2} H_n(t)."""
    return np.exp(-2 * V * t) * np.exp((R / V) * ((1 - np.exp(-V * t))))