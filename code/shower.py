__author__ = 'Fiona C. Pärli'
__email__ = 'fiona.paerli@students.unibe.ch'
__date__ = 'June 2025'

import numpy as np
from four_vector import FourVector as FV
from typing import Callable
from transform import Transform
from plotting import Hist, Hist2D
from itertools import combinations
import time
import recoil
import os
import matplotlib.pyplot as plt
import event_analysis
from event_analysis import energy_out
from athrust_accel import athrust  # Import the Rust-accelerated function
import sys
from tqdm import tqdm

N_c = 3
cutMassive = 1e-7
cutInfRap = 1e-8
cutBoost = 1e-10
toleranceSameVector = 1e-10
global cut
cut = 4

M_Z = 91.2  # Z boson mass in GeV
IAS = 106.495  # Inverse strong coupling constant approximation
T_STEP = 0.005  # Bin size for histograms

N_VEC = FV(1, 0, 0, 1)*M_Z/2
N_BAR = FV(1, 0, 0, -1)*M_Z/2

shower_output_dir = "shower_output"
os.makedirs(shower_output_dir, exist_ok=True)

def virtual_correction(n1: FV, n2: FV): 
    """compute the virtual correction

    Args:
        n1 (FV): first parent vector of the emitting dipole
        n2 (FV): second parent vector of the emitting dipole

    Returns:
        float: virtual correction
    """
    n1 = n1/FV.energy(n1)
    n2 = n2/FV.energy(n2)
    M2 = 2*FV.four_dot(n1, n2)
    epsilonEta = cut
    epsilon = np.sqrt(1-np.tanh(epsilonEta))
    epsq= epsilon*epsilon
    alpha = (M2 - 2*epsq)/(2*epsq)
    beta = np.sqrt(1 - M2/4)
    res = np.log(beta + np.sqrt(abs(alpha+ beta**2)))
    
    return 2*res

def GenRandV(n1: FV, n2: FV):
    """Boosts into the back-to-back frame of n1, n2 and generates a new random vector

    Args:
        n1 (FV): first parent vector of the emitting dipole
        n2 (FV): second parent vector of the emitting dipole

    Returns:
        FV: new vector, the emission
    """
    global nan_bool
    if np.isnan(FV.to_array(n1)).any() or np.isnan(FV.to_array(n2)).any():
        print("OOO")
        print(n1, n2)
        nan_bool = True
    n1 = n1/FV.energy(n1)
    n2 = n2/FV.energy(n2)
    M2 = 2 * FV.four_dot(n1, n2)
    M = np.sqrt(M2)
    beta = np.sqrt(1 - M2 / 4)
    epsilonEta = cut
    epsilon = np.sqrt(1-np.tanh(epsilonEta))
    epsq= epsilon*epsilon
    alpha = (M2 - 2 * epsq) / (2 * epsq)
    phiMax = np.pi
    phi = phiMax * (-1 + 2 * np.random.random_sample())
    sip = np.sin(phi)
    cop = np.cos(phi)
    etaMax = np.log(beta + np.sqrt(alpha + beta**2))
    eta = -etaMax + 2 * etaMax * np.random.random_sample()
    shy = 1 / np.cosh(eta)
    thy = np.tanh(eta)

    newvec = FV(1.0, (beta-shy*cop)/(1-beta*shy*cop), M/2*shy*sip/(1-beta*shy*cop),
                                                           M/2*thy/(1-beta*shy*cop))

    n1p=np.array([1,beta,0,M/2])
    n2p=np.array([1,beta,0,-M/2])
    vvec = np.array([1, 0, 0, 0])
    metric = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])
    n1 = FV.to_array(n1)
    n2 = FV.to_array(n2)
    
    def boost(v,vp):
        ''' boost vector vp into v'''
        diff = vp-v
        ndiff= sp(diff,diff)
        boostm=  np.identity(4)
        #if abs(ndiff) < 1e-10 :
        #    print('no: ',v,' and',vp)
        if abs(ndiff) > 1.0e-10 :
            boostm += -2/ndiff*np.tensordot(diff,np.dot(metric,diff),0)
        return boostm

    def sp(v1,v2):
        ''' returns the scalar product of four vectors v1,v2''' 
        return np.dot(v1,np.dot(metric,v2))
    
    # Boost n1p --> n1
    invboost=boost(n1,n1p)
    n2pp=np.dot(invboost,n2p)
    # Rotate n2pp into n2 without changing n1
    n2pPerp=n2pp-vvec-(1-M2/2)*(n1-vvec)
    n2Perp=n2-vvec-(1-M2/2)*(n1-vvec)
    rotation=boost(n2Perp,n2pPerp)
    
    newvec = FV.to_array(newvec)
    newvec=np.dot(rotation,np.dot(invboost,newvec))

    return FV(*newvec)

def shower(origEvent: list[FV], NeV: int, tstart: float, tmax: float,
           initialweight: float, recoil_on: bool, output_file: str,
           angle_in_degrees: float, recoil_function: Callable, IAS: float = IAS,
           baseline_bin_E=None, baseline_bin_t=None,
           baseline_E_out=None, baseline_t_out=None):
    """
    Run a parton shower simulation with optional recoil and Sudakov factor computation.

    The function saves all histograms as PDF plots in the `histograms` folder.
    If `recoil_on` is False, a default collinear emission with no recoil is applied.

    After the evolution, this function produces histograms for:
        - $S(t)$: Sudakov factor in shower time.
        - $S(E)$: Sudakov factor in energy.
        - Out-of-cone energy probability density $P(E_{\text{out}})$.
        - A 2D histogram of $(t, t_E)$ to compare shower time and energy-equivalent time.
        - Comparison plots for different recoil schemes and baseline results.

    Args:
        origEvent (list[FV]): Initial list of four-momenta representing the hard event.
        NeV (int): Number of shower runs (Monte Carlo events).
        tstart (float): Initial shower time.
        tmax (float): Maximum shower evolution time.
        initialweight (float): Starting event weight for normalization.
        recoil_on (bool): Whether to apply the specified recoil scheme.
        output_file (str): Path prefix for output files (unused directly here, kept for compatibility).
        angle_in_degrees (float): Cone angle to test if emissions escape.
        recoil_function (Callable): Function defining the recoil scheme (e.g., asymmetric, symmetric, PanGlobal).
        baseline_bin_E (Hist, optional): Baseline $S(E)$ histogram for comparison plots.
        baseline_bin_t (Hist, optional): Baseline $S(t)$ histogram for comparison plots.
        baseline_E_out (Hist, optional): Baseline $E_{\text{out}}$ histogram for comparison.
        baseline_t_out (Hist, optional): Baseline $t_{\text{out}}$ histogram for comparison.

    Returns:
        - bin_S_of_E (Hist): Histogram of $S(E)$.
        - bin_S_of_t (Hist): Histogram of $S(t)$.
        - hist_E_outside (Hist): Histogram of $E_{\text{out}}$ (emissions breaking the cone).
        - hist_t_outside (Hist): Histogram of $t_{\text{out}}$ for emissions outside the cone.
        - hist_E_outside_x3bins (Hist): Same as `hist_E_outside` but using cube-root binning.
    """

    n_broken = 0

    bin_S_of_t = Hist(20, 0.1)
    hist_t_outside = Hist(20, 0.1)
    hist_tE_outside = Hist(20, 0.1)

    bin_S_of_E = Hist(20, M_Z/2)
    hist_E_outside = Hist(20, M_Z/2)
    hist_E_outside_x3bins = Hist(20, M_Z/2, x3bin=True)

    hist_t_tE_2D = Hist2D(20, 20, 0.1, 0.1)

    # Monte Carlo loop
    for _ in tqdm(range(0, NeV), desc="Running Monte Carlo Shower", ncols=100, unit="event"):
        event = list(origEvent)
        virtuals = [virtual_correction(event[n - 1], event[n]) for n in range(1, len(event))]
        virtualtot = sum(virtuals)
        norm = 4 * N_c * virtualtot * NeV
        w = float(initialweight) / norm
        t = tstart
        E_max = np.sum([FV.energy(vec) for vec in origEvent])/2

        while t < tmax:
            dt = -np.log(np.random.random_sample()) / (4 * N_c * virtualtot)
            t += dt

            # fill inside-cone Sudakov
            bin_S_of_t.add_to_bin(t, w)

            #fill inside-cone Sudakov in energy
            En = E_max*np.exp(-IAS*t)
            jac = 1/(En*IAS)
            bin_S_of_E.add_to_bin(En, w*jac)
            
            # select dipole
            virtsum = np.cumsum(virtuals) / virtualtot
            
            position = np.searchsorted(virtsum, np.random.random_sample(), side = "right")
            n3 = GenRandV(event[position], event[position + 1])
            q1, q2 = event[position], event[position + 1]

            # kinematics
            if recoil_on:
                args = {
                    recoil.asymmetric_recoil: (q1, q2, n3, dt),
                    recoil.symmetric_recoil: (q1, q2, n3, dt),
                    recoil.panglobal_shower_dipole: (q1, q2, n3, event, dt, position),
                }[recoil_function]
                if recoil_function is recoil.panglobal_shower_dipole:
                    q3, event = recoil_function(*args)
                else:
                    q1_new, q2_new, q3 = recoil_function(*args)
                    event = event[:position] + [q1_new, q3, q2_new] + event[position + 2:]
                thrust_val, thrust_vec = athrust(np.array([v.spatial_part() for v in event], dtype=np.float64))
            else:
                energy_scale = np.exp(-IAS * dt)
                denom = FV.four_dot(q1, n3) + FV.four_dot(q2, n3)
                q3 = energy_scale * FV.four_dot(q1, q2) / denom * n3 / FV.energy(n3)
                event = event[:position] + [q1, q3, q2] + event[position + 2:]
                thrust_vec = np.array([0.0, 0.0, 1.0])

            eout = 0
            for vec in event:
                eout += energy_out(vec, thrust_vec, angle_in_degrees)
            
            if eout != 0:
                n_broken += 1
                hist_E_outside.add_to_bin(eout, 1)
                hist_E_outside_x3bins.add_to_bin(eout, weight=1)
                hist_t_outside.add_to_bin(t, 1)

                tE_out = (1/IAS)*np.log(E_max/eout)
                hist_tE_outside.add_to_bin(tE_out, 1)

                hist_t_tE_2D.add(t, tE_out, 1/NeV)
                break

            # update virtuals
            virtuals=[ virtual_correction(event[n-1],event[n]) for n in range(1,len(event)) ]

            newvirt = sum(virtuals)
            w *= virtualtot / newvirt
            virtualtot = newvirt

    histogram_dir = "histograms"
    recoil_name = recoil_function.__name__ if recoil_on else ""
    recoil_str = "recoil" if recoil_on else "norecoil"
    label = f"{angle_in_degrees}°"

    bin_S_of_E.plot_histograms_energy(
        [bin_S_of_E],
        filename=os.path.join(histogram_dir, f"S(E)_{recoil_str}_({recoil_name})"),
        xlabel=r"$E/E_{\max}$",
        NeV=NeV,
        ylabel=r"$S(E)$",
        title="Sudakov Form Factor $S(E)$",
        cumulative=[True],
        survival=[True],
        normalize_energy_axis=True,
    )

    hist_E_outside.plot_histograms_energy(
        [hist_E_outside],
        filename=os.path.join(histogram_dir, f"S(E)_outside_{recoil_str}_({recoil_name})"),
        xlabel=r"$E_{\text{out}} / E_{\max}$",
        NeV=NeV,
        ylabel=r"$P(E_{\text{out}})$",
        title="Emission Energy $E_{out}$ Breaking the Cone",
        cumulative=[False]*(1 + int(baseline_E_out is not None)),
        survival=[False]*(1 + int(baseline_E_out is not None)),
        labels=(["No Recoil", label] if (recoil_on and baseline_E_out) else [label]),
        normalize_energy_axis=True,
        styles=(['dotted', 'solid'] if (recoil_on and baseline_E_out) else ['solid']), 
        ylog = True
    )

    hist_E_outside_x3bins.plot_histograms_energy(
        [hist_E_outside_x3bins],
        filename=os.path.join(histogram_dir, f"S(E)_outside_{recoil_str}_({recoil_name})_x3bins"),
        xlabel=r"prbability density $(E_{\text{out}} / E_{\max})^{1/3}$",
        ylabel=0,
        title="Out-of-Cone Spectrum, Cube-Root Binning",
        NeV=NeV,
        cumulative=[False],
        survival=[False],
        normalize_energy_axis=True
    )

    both_t_hists = Hist(20, 0.1)
    both_t_hists.plot_histograms(
        [bin_S_of_t, hist_t_outside, hist_tE_outside], 
        filename=os.path.join(histogram_dir, f"S(t)_S(tE)_{recoil_str}_{recoil_name}"),
        xlabel=r"$t$ or $t_E$", ylabel=r"$S(t)$ or $S(t_E)$",
        NeV = NeV,
        title="S(t) both methods and S(t_E) derivative",
        cumulative=[False, True, True], survival=[False, True, True],
        labels=["S(t) using method 1: direct", "S(t) method 2: derivative", "S(t_E) method 2: derivative"], 
        normalize_energy_axis=True
    )

    hist_t_tE_2D.plot_density(
        filename=os.path.join(histogram_dir, f"2D_t_tE_density_{recoil_str}_{recoil_name}.pdf"),
        xlabel="t",
        ylabel=r"$t_E$"
    )

    hist_t_tE_2D.plot_expectation(
        filename=os.path.join(histogram_dir, f"2D_t_tE_expectation_{recoil_str}_{recoil_name}.pdf"),
        xlabel="t",
        ylabel=r"$t_E$"
    )

    # combined comparison plots
    S_t_hist = Hist(20, 0.1)
    print("Here", bin_S_of_t.entries, hist_t_outside.entries)
    S_t_hist.plot_histograms(
        [bin_S_of_t, hist_t_outside],
        filename=os.path.join(histogram_dir, f"S(t)_both_methods_{recoil_str}_{recoil_name}"),
        xlabel="t", ylabel="S(t)",
        NeV = NeV,
        title="S(t) using both methods",
        cumulative=[False, True], survival=[False, True],
        labels=["Method 1: direct", "Method 2: derivative"], 
        normalize_energy_axis=True
    )

    #check to see if hist_E_outside makes sense
    bin_edges = hist_E_outside.bins
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    density = hist_E_outside.entries / (bin_widths * NeV)
    total_prob = np.sum(hist_E_outside.entries) / NeV
    print(f"Estimated probability of an event breaking the cone: {total_prob:.6f}")
    # Should give same answer:
    prob_from_integral = np.sum(density * bin_widths)
    print(f"Integrated PDF area: {prob_from_integral:.6f}")
    # Should give same answer again: 
    print(f"Direct fraction: {n_broken / NeV:.6f}")

    return bin_S_of_E, bin_S_of_t, hist_E_outside, hist_t_outside, hist_E_outside_x3bins
