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
import plotting

#This file contains the same content as the shower.py file, except it has the running coupling (instead of fixed coupling at alpha_s(M_Z))

N_c = 3
cutMassive = 1e-7
cutInfRap = 1e-8
cutBoost = 1e-10
toleranceSameVector = 1e-10
global cut
cut = 4

M_Z = 91.2  # Z boson mass in GeV
T_STEP = 0.005  # Bin size for histograms

N_VEC = FV(1, 0, 0, 1)*M_Z/2
N_BAR = FV(1, 0, 0, -1)*M_Z/2

IAS_MZ = 106.495
n_f = 5
beta_0 = (11*N_c - 2*n_f)/3

def alpha_s(Q: float):
    """compute the running coupling

    Args:
        Q (float): energy scale

    Returns:
        float: alpha_s(Q) value of the running coupling
    """
    Q = max(Q, 1.0) 
    alpha_s_MZ = 4*np.pi/IAS_MZ
    
    denominator = 1/alpha_s_MZ + (beta_0/(4*np.pi)) * np.log(Q**2 / M_Z**2)
    
    return 1 / denominator

shower_output_dir = "shower_output"
os.makedirs(shower_output_dir, exist_ok=True)

#this code only works for massless particles!!! 
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
           angle_in_degrees: float, recoil_function: Callable,
           baseline_bin_E=None, baseline_bin_t=None,
           baseline_E_out=None, baseline_t_out=None):
    """
    Run a parton shower simulation with optional recoil schemes and running coupling.

    This function implements a dipole-based parton shower Monte Carlo. Starting from an initial 
    list of four-momenta, it evolves the event in shower time, generating emissions according to 
    virtual corrections and the Sudakov form factor. The QCD coupling can run dynamically with 
    the dipole invariant mass.

    If a recoil scheme is enabled, it updates the kinematics of the emitter and spectator 
    according to the specified recoil function (e.g., asymmetric, symmetric, PanLocal, PanGlobal).
    For each emission, the function checks if particles exit a cone defined by `angle_in_degrees`
    relative to the thrust axis.

    Multiple histograms are filled:
        - `S(t)` from direct integration over shower time.
        - `S(E)` from integration over the energy scale with running coupling.
        - Probability density of out-of-cone emissions, `P(E_out)`.
        - Alternative binning for `E_out` (cube-root bins).
        - A 2D correlation histogram of `t` and `t_E` (the time derived from out-of-cone energy).

    All histograms are saved as plots in `histograms`. Comparison plots for baselines 
    can be produced if they are supplied. 
    Uses a running $\alpha_s(Q)$ with $Q$ set by the dipole invariant mass.
    Diagnostic printouts show the total probability for an emission to break the cone,
    compared by direct count and by PDF integral.
    This routine assumes the `Hist` and `Hist2D` classes have .plot_histograms_* and
    .plot_density/.plot_expectation methods.

    Args:
        origEvent (list[FV]): Initial four-momenta representing the Born-level hard event.
        NeV (int): Number of Monte Carlo shower runs.
        tstart (float): Starting shower evolution time.
        tmax (float): Maximum shower evolution time.
        initialweight (float): Initial weight for event normalization.
        recoil_on (bool): If True, apply the specified recoil scheme.
        output_file (str): Path for output files (not used internally here).
        angle_in_degrees (float): Cone angle to test for out-of-cone emissions.
        recoil_function (Callable): Selected recoil routine, e.g. asymmetric, symmetric, PanLocal, PanGlobal.
        baseline_bin_E (Hist, optional): Baseline Sudakov $S(E)$ for comparison.
        baseline_bin_t (Hist, optional): Baseline Sudakov $S(t)$ for comparison.
        baseline_E_out (Hist, optional): Baseline $E_{out}$ histogram for comparison.
        baseline_t_out (Hist, optional): Baseline $t_{out}$ histogram for comparison.

    Returns:
        - bin_S_of_E (Hist): Histogram of Sudakov factor $S(E)$.
        - bin_S_of_t (Hist): Histogram of Sudakov factor $S(t)$.
        - hist_E_outside (Hist): Out-of-cone energy probability $P(E_{out})$.
        - hist_t_outside (Hist): Distribution of shower times where out-of-cone emissions occur.
        - hist_E_outside_x3bins (Hist): Out-of-cone energy histogram with cube-root binning.
    """

    n_broken = 0
    bin_S_of_t = Hist(20, 0.1)
    hist_t_outside = Hist(20, 0.1)
    hist_tE_outside = Hist(20, 0.1)

    bin_S_of_E = Hist(20, M_Z/2)
    hist_E_outside = Hist(20, M_Z/2)
    hist_E_outside_x3bins = Hist(20, M_Z/2, x3bin=True)

    hist_t_tE_2D = Hist2D(20, 20, 0.1, 0.1)

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

            bin_S_of_t.add_to_bin(t, w)

            virtsum = np.cumsum(virtuals) / virtualtot
            position = np.searchsorted(virtsum, np.random.random_sample(), side="right")
            q1, q2 = event[position], event[position + 1]

            Q_scale = max(np.sqrt(abs(2 * FV.four_dot(q1, q2))), 1)
            as_run = alpha_s(Q_scale)
            IAS_run = 1/ as_run

            En = E_max * np.exp(-IAS_run * t)
            jac = 1/(En * IAS_run)

            bin_S_of_E.add_to_bin(En, w * jac)

            n3 = GenRandV(q1, q2)

            if recoil_on:
                args = {
                    recoil.asymmetric_recoil: (q1, q2, n3, dt, IAS_run),
                    recoil.symmetric_recoil: (q1, q2, n3, dt, IAS_run),
                    recoil.panglobal_shower_dipole: (q1, q2, n3, event, dt, position, IAS_run),
                }[recoil_function]
                if recoil_function is recoil.panglobal_shower_dipole:
                    q3, event = recoil_function(*args)
                else:
                    q1_new, q2_new, q3 = recoil_function(*args)
                    event = event[:position] + [q1_new, q3, q2_new] + event[position + 2:]
                thrust_val, thrust_vec = athrust(np.array([v.spatial_part() for v in event], dtype=np.float64))
            else:
                energy_scale = np.exp(-IAS_run * dt)
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
                tE_out = (1/IAS_run) * np.log(E_max / eout)
                hist_tE_outside.add_to_bin(tE_out, 1)

                hist_t_tE_2D.add(t, tE_out, 1/NeV)
                break

            virtuals = [virtual_correction(event[n - 1], event[n]) for n in range(1, len(event))]
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
        xlabel=r"$E/E_{max}$",
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
        xlabel=r"$E_{\text{out}} / E_{max}$",
        NeV=NeV,
        ylabel=r"$P(E_{\text{out}})$",
        title="Emission Energy $E_{out}$ Breaking the Cone",
        cumulative=[False]*(1 + int(baseline_E_out is not None)),
        survival=[False]*(1 + int(baseline_E_out is not None)),
        labels=(["No Recoil", label] if (recoil_on and baseline_E_out) else [label]),
        normalize_energy_axis=True,
        styles=(['dotted', 'solid'] if (recoil_on and baseline_E_out) else ['solid']),
        ylog=True
    )

    hist_E_outside_x3bins.plot_histograms_energy(
        [hist_E_outside_x3bins],
        filename=os.path.join(histogram_dir, f"S(E)_outside_{recoil_str}_({recoil_name})_x3bins"),
        xlabel=r"prbability density $(E_{\text{out}} / E_{max})^{1/3}$",
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
        NeV=NeV,
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

    S_t_hist = Hist(20, 0.1)
    print("Here", bin_S_of_t.entries, hist_t_outside.entries)
    S_t_hist.plot_histograms(
        [bin_S_of_t, hist_t_outside],
        filename=os.path.join(histogram_dir, f"S(t)_both_methods_{recoil_str}_{recoil_name}"),
        xlabel="t", ylabel="S(t)",
        NeV=NeV,
        title="S(t) using both methods",
        cumulative=[False, True], survival=[False, True],
        labels=["Method 1: direct", "Method 2: derivative"],
        normalize_energy_axis=True
    )

    bin_edges = hist_E_outside.bins
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    density = hist_E_outside.entries / (bin_widths * NeV)
    total_prob = np.sum(hist_E_outside.entries) / NeV
    print(f"Estimated probability of an event breaking the cone: {total_prob:.6f}")

    prob_from_integral = np.sum(density * bin_widths)
    print(f"Integrated PDF area: {prob_from_integral:.6f}")
    print(f"Direct fraction: {n_broken / NeV:.6f}")

    return bin_S_of_E, bin_S_of_t, hist_E_outside, hist_t_outside, hist_E_outside_x3bins

def run_shower_test(NeV: int, tmax: float, angle_in_degrees: float):
    """run shower with running coupling

    Args:
        NeV (int): Number of events
        tmax (float): Maximum time evolution variable t
        angle_in_degrees (float): Cone opening angle
    """
    orig_event = [FV(M_Z/2, 0, 0, M_Z/2), FV(M_Z/2, 0, 0, -M_Z/2)]

    # 1) Run no-recoil baseline
    baseline_S_E, baseline_S_t, baseline_E_out, baseline_t_out, baseline_E_out_x3bins = shower(
        origEvent=orig_event,
        NeV=NeV,
        tstart=0.0,
        tmax=tmax,
        initialweight=1.0,
        recoil_on=False,
        output_file="shower_output_norecoil.txt",
        angle_in_degrees=angle_in_degrees,
        recoil_function=None
    )

    from recoil import asymmetric_recoil, symmetric_recoil, panglobal_shower_dipole
    # 2) Run each recoil scheme
    recoil_funcs = [asymmetric_recoil, symmetric_recoil, panglobal_shower_dipole]
    S_E_all = [baseline_S_E]
    E_out_all = [baseline_E_out]
    labels = ["No Recoil"]

    for recoil_func in recoil_funcs:
        S_E, _, E_out, _, _ = shower(
            origEvent=orig_event,
            NeV=NeV,
            tstart=0.0,
            tmax=tmax,
            initialweight=1.0,
            recoil_on=True,
            output_file=f"shower_output_{recoil_func.__name__}.txt",
            angle_in_degrees=angle_in_degrees,
            recoil_function=recoil_func
        )
        S_E_all.append(S_E)
        E_out_all.append(E_out)
        labels.append(recoil_func.__name__)

    # 3) Combined S(E) plot
    S_E_all[0].plot_histograms_energy(
        S_E_all,
        filename=os.path.join("histograms", "combined_S_E_all.pdf"),
        xlabel=r"$E/E_{\max}$",
        ylabel=r"$S(E)$",
        NeV=NeV,
        title="Sudakov $S(E)$ for All Recoil Schemes",
        cumulative=[True] * len(S_E_all),
        survival=[True] * len(S_E_all),
        labels=labels,
        normalize_energy_axis=True
    )

    # 4) Combined E_out plot
    E_out_all[0].plot_histograms_energy(
        E_out_all,
        filename=os.path.join("histograms", "combined_E_out_all.pdf"),
        xlabel=r"$E_{\text{out}} / E_{\max}$",
        ylabel=r"$p(E_{\text{out}})$",
        NeV=NeV,
        title=r"Out-of-Cone $E_{\text{out}}$ for All Recoil Schemes",
        cumulative=[False] * len(E_out_all),
        survival=[False] * len(E_out_all),
        labels=labels,
        normalize_energy_axis=True,
        ylog=True
    )

if __name__ == "__main__":
    plotting.clear_directory("histograms")

    NeV = 10_000 
    tmax = 0.1
    angle_in_degrees = 60.0

    run_shower_test(NeV, tmax, angle_in_degrees)
