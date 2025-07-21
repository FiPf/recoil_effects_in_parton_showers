import numpy as np
from four_vector import FourVector as FV
from typing import Callable
import transform
import shower

M_Z = 91.2  # Z boson mass in GeV
IAS_MZ = 106.495
alpha_MZ = 1/IAS_MZ

def asymmetric_recoil(q1: FV, q2: FV, n3: FV, t: float, ias: float = IAS_MZ):
    """asymmetric recoil scheme, locally redistributes the recoil among the two parent vectors of the dipole
    one vector is the spectator, one is the emitter

    Args:
        q1 (FV): parent vector 1 from the dipole
        q2 (FV): parent vector 2 from the dipole
        n3 (FV): normalized vector of the emission
        t (float): evolution time variable of the emission
        ias (float, optional): Inverse of the coupling constant. Defaults to IAS_MZ.

    Returns:
        q1_new (FV): recoil adapted parent vector 1 from the dipole
        q2_new (FV): recoil adapted parent vector 2 from the dipole
        q3 (FV): vector of the emission with the right energy
    """
    energy_scale = np.exp(-ias * t) #Computes the energy of the new emission along direction n3.

    denominator = FV.four_dot(q1, n3) + FV.four_dot(q2, n3)
    q3 = energy_scale*FV.four_dot(q1, q2)/denominator*n3

    y123 = FV.four_dot(q1, q3) / FV.four_dot(q1 - q3, q2)
    y213 = FV.four_dot(q2, q3) / FV.four_dot(q1, q2 - q3)

    if y123 < y213:
        y = y123
        q1_new = q1 + q2 * y - q3
        q2_new = q2 * (1 - y)
    else:
        y = y213
        q1_new = q1 * (1 - y)
        q2_new = q2 + q1 * y - q3

    if y > 1 or y < 0:
        print(f"ERROR: Invalid y value: {y}")

    if np.isnan(FV.to_array(q1_new)).any() or np.isnan(FV.to_array(q2_new)).any() or np.isnan(FV.to_array(q3)).any():
        print("new recoil")
        print("q1_new:", q1_new)
        print("q2_new:", q2_new)
        print("q3:", q3)

    total_momentum_before = q1 + q2
    total_momentum_after = q1_new + q2_new + q3

    if not np.allclose(FV.to_array(total_momentum_before), FV.to_array(total_momentum_after)):
        print("ERROR: Momentum conservation is violated!")
        print(f"Total momentum before: {total_momentum_before}")
        print(f"Total momentum after: {total_momentum_after}")

    return q1_new, q2_new, q3

def symmetric_recoil(q1: FV, q2: FV, n3: FV, dt: float, ias: float = IAS_MZ):
    """symmetric recoil scheme, locally redistributes the recoil among the two parent vectors of the dipole
    recoil gets shared equally

    Args:
        q1 (FV): parent vector 1 from the dipole
        q2 (FV): parent vector 2 from the dipole
        n3 (FV): normalized vector of the emission
        dt (float): evolution time variable of the emission
        ias (float, optional): Inverse of the coupling constant. Defaults to IAS_MZ.

    Returns:
        q1_new (FV): recoil adapted parent vector 1 from the dipole
        q2_new (FV): recoil adapted parent vector 2 from the dipole
        q3 (FV): vector of the emission with the right energy
    """
    energy_scale = np.exp(-ias * dt)
    factor = FV.four_dot(q1, q2) / (FV.four_dot(q1, n3) + FV.four_dot(q2, n3))
    q3 = (n3*factor/ n3.energy()) * energy_scale

    import sympy as sp
    alpha, beta = sp.symbols('alpha beta', real=True)
    q1q2, q1q3, q2q3 = sp.symbols('q1q2 q1q3 q2q3', real=True)

    q1_new_sq = (2 * beta * (1 - alpha) * q1q2 -(1 - alpha) * q1q3 - beta * q2q3)
    q2_new_sq = (2 * alpha * (1 - beta) * q1q2 - alpha * q1q3 - (1 - beta) * q2q3)
    eq1 = sp.Eq(q1_new_sq, 0)
    eq2 = sp.Eq(q2_new_sq, 0)

    solution = sp.solve([eq1, eq2], (alpha, beta), dict=True)
    sol = solution[0]  # take the first solution
    al = sol[alpha]
    be = sol[beta]

    q1q2_val = FV.four_dot(q1, q2)
    q1q3_val = FV.four_dot(q1, q3)
    q2q3_val = FV.four_dot(q2, q3)
    subs = {q1q2: q1q2_val, q1q3: q1q3_val, q2q3: q2q3_val}
    al = float(al.evalf(subs=subs))
    be = float(be.evalf(subs=subs))

    q1_new = (1-al)*q1 + be*q2 - 0.5*q3
    q2_new = al*q1 + (1-be)*q2 - 0.5*q3

    total_momentum_before = q1 + q2
    total_momentum_after = q1_new + q2_new + q3

    if not np.allclose(FV.to_array(total_momentum_before), FV.to_array(total_momentum_after)):
        print("ERROR: Momentum conservation is violated!")
        print(f"Total momentum before: {total_momentum_before}")
        print(f"Total momentum after: {total_momentum_after}")

    for name, v in [("q1_new", q1_new), ("q2_new", q2_new), ("q3", q3), ("q1", q1), ("q2", q2)]:
        m2 = FV.four_dot(v, v)
        if not np.isclose(m2, 0.0, atol=1e-6):
            print(f"ERROR: {name} has nonzero mass² = {m2:.3e}")

    return q1_new, q2_new, q3

def apply_panglobal_boost(p: FV, Q: FV, Q_bar: FV):
    """Apply the PanGlobal recoil boost to a single four-vector.
    This constructs the boost matrix that maps total momentum Q to Q_bar,
    and applies it to the given FourVector p.

    Args:
        p (FV): the FourVector to be boosted. 
        Q (FV): Original total four-momentum before emission.
        Q_bar (FV): Total four-momentum after emission and local recoil.

    Returns:
        FV: The boosted four-vector 
    """
    B = construct_panglobal_boost(Q, Q_bar)
    return FV(*B@p.to_array())

def construct_panglobal_boost(Q: FV, Q_bar: FV): 
    """Check lengths and lightlikeness before constructing boost

    Args:
        Q (FV): Original total four-momentum before emission.
        Q_bar (FV): Total four-momentum after emission and local recoil.

    Raises:
        ValueError: Q and Q_bar must have the same Minkowski length

    Returns:
        np.array: 4x4 boost matrix
    """
    if not check_same_length(Q, Q_bar):
        raise ValueError("Q and Q_bar must have the same Minkowski length.")
    H1 = transform.householder_from_normal(Q + Q_bar)
    H2 = transform.householder_from_normal(Q)
    B = H2 @ H1
    return B 

def check_same_length(Q: FV, Qp: FV, tol: float=1e-8) -> bool:
    """Check whether two four-vectors have the same Minkowski length
    within a numerical tolerance. This is required when constructing a valid Lorentz boost
    to avoid unphysical scaling.

    Args:
        Q (FV): Original total four-momentum before emission.
        Q_bar (FV): Total four-momentum after emission and local recoil.
        tol (float, optional): Tolerance. Defaults to 1e-8.

    Returns:
        bool: True if lengths match within tolerance.
    """
    len_Q = Q.four_dot(Q)
    len_Qp = Qp.four_dot(Qp)
    same_length = abs(len_Q - len_Qp) < tol
    if not same_length:
        print(f"Length mismatch: |Q² - Qp²| = {abs(len_Q - len_Qp)}")
    return same_length

def panglobal_shower_dipole(q1: FV, q2: FV, n3: FV, event: list[FV], dt: float, position: int, ias: float = IAS_MZ):
    """PanGlobal recoil scheme, globally redistribute the recoil

    Args:
        q1 (FV): parent vector 1 from the dipole
        q2 (FV): parent vector 2 from the dipole
        n3 (FV): normalized vector of the emission
        event (list[FV]): list of all FourVectors of that event
        dt (float): evolution time variable
        position (int): index of the emitting dipole
        ias (float, optional): Inverse of coupling constant. Defaults to IAS_MZ.

    Returns:
        FV: FourVector of the emitted particle after boost.
        list[FV]: Updated list of event four-vectors after boost.
    """
    energy_scale = np.exp(-ias * dt)

    # Calculate max allowed energy for emission along n3 direction
    E3_max = min(FV.four_dot(q1, q2) / FV.four_dot(n3, q1),
                 FV.four_dot(q1, q2) / FV.four_dot(n3, q2))
    q3 = energy_scale * E3_max * n3  # emitted particle 4-vector

    # Recoil coefficients
    a_k = FV.four_dot(q2, q3) / FV.four_dot(q1, q2)
    b_k = FV.four_dot(q1, q3) / FV.four_dot(q1, q2)

    # Check if coefficients are physical
    if a_k > 1 or b_k > 1:
        print(f"Warning: recoil coefficients > 1 (a_k={a_k}, b_k={b_k})")

    # Update spectator and emitter momenta
    p1_new = q1 * (1 - a_k)
    p2_new = q2 * (1 - b_k)

    # Replace old vectors in event with new recoiled ones plus emitted particle
    new_event = event[:position] + [p1_new, q3, p2_new] + event[position + 2:]

    # Momentum before boost (sum of four-vectors)
    Qvalue = 91.2  # reference scale
    Qtot = FV(Qvalue, 0, 0, 0)
    Qbar = sum(new_event, FV(0, 0, 0, 0))

    # Compute rescale factor to conserve energy scale
    Qbar_mag = np.sqrt(FV.four_dot(Qbar, Qbar))
    rscale = Qvalue / Qbar_mag

    # Rescale all momenta to match scale
    new_event = [p * rscale for p in new_event]

    # Construct boost matrix from Qtot to new total momentum Qbar
    Qprime = sum(new_event, FV(0,0,0,0))


    boost_matrix = construct_panglobal_boost(Qtot, Qprime)

    # Apply boost to all four-vectors
    boosted_event = [FV.matmul(p, boost_matrix) for p in new_event]

    # The emitted particle after boost is at position + 1
    p_k_after = boosted_event[position + 1]

    return p_k_after, boosted_event
