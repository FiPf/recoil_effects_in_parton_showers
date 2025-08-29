__author__ = 'Fiona C. Pärli'
__email__ = 'fiona.paerli@students.unibe.ch'
__date__ = 'June 2025'

import numpy as np
from four_vector import FourVector as FV
from typing import Callable
from transform import Transform
from plotting import Hist
import recoil
import os
import matplotlib.pyplot as plt

#chose which version you want
from athrust_accel import athrust  # Import the Rust-accelerated function
#from thrust import athrust #Python thrust function, slow

from tqdm import tqdm
import pandas as pd
import event_analysis

#This file contains the same content as the shower.py file, except it makes the gap fraction heatmap

N_c = 3
cutMassive = 1e-7
cutInfRap = 1e-8
cutBoost = 1e-10
toleranceSameVector = 1e-10
cut = 4

M_Z = 91.2  # Z boson mass in GeV
IAS = 106.495  # Inverse strong coupling constant approximation
T_STEP = 0.005  # Bin size for histograms

N_VEC = FV(1, 0, 0, 1)*M_Z/2
N_BAR = FV(1, 0, 0, -1)*M_Z/2

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

def GenRandV(n1, n2):
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

    return FV(*newvec)#Transform.apply_boost_and_rotation(n1, n2, newvec)

def shower(origEvent: list[FV], NeV: int, tstart: float, tmax: float,
           initialweight: float, recoil_on: bool, output_file: str,
           angle_in_degrees: float, recoil_function: Callable):
    """
    Simulate a parton shower evolution and record full emission data to a CSV file.

    At each emission step, the function records:
        - The current evolution variable `t`
        - The current event momenta
        - The thrust value and thrust axis (if recoil is on)
        - The shower weight `w`
        - The emission index and run index

    All events and emissions are collected in a DataFrame and written to the specified CSV `output_file`.

    Parameters
    ----------
    origEvent : list[FV]
        List of initial FourVector objects representing the starting event.
    NeV : int
        Number of Monte Carlo shower runs to perform.
    tstart : float
        Starting value of the evolution variable `t`.
    tmax : float
        Maximum value of the evolution variable `t`. The shower stops once `t` reaches this.
    initialweight : float
        Initial weight assigned to the shower; updated according to Sudakov factors.
    recoil_on : bool
        If True, apply recoil kinematics using `recoil_function`. If False, use no-recoil scheme.
    output_file : str
        Path to the CSV file where the full shower event record will be written.
    angle_in_degrees : float
        Cone angle (in degrees) used to compute out-of-cone emissions and thrust axis.
    recoil_function : Callable
        The recoil scheme function to apply when `recoil_on` is True.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the recorded emissions for all shower runs.
        Columns include:
            - run_id: the shower run index
            - emission_id: the index of the emission in that run
            - t: evolution variable value
            - w: event weight at that step
            - event: list of four-momenta after the emission
            - thrust_val: computed thrust (if recoil_on is True)
            - thrust_vec: thrust axis vector (if recoil_on is True)

    Notes
    -----
    - Virtual corrections are recalculated at each step.
    - The output can be post-processed to compute Sudakov form factors, gap fractions, or energy flow observables.
    """

    records = []

    for x in tqdm(range(1, NeV), desc="Running Monte Carlo Shower", ncols=100, unit="event"):
        event = list(origEvent)
        # initial virtual corrections
        virtuals = [virtual_correction(event[n - 1], event[n]) for n in range(1, len(event))]
        virtualtot = sum(virtuals)
        norm = 4 * N_c * virtualtot * NeV
        w = float(initialweight) / norm
        t = tstart
        emissions = 0
        first_emission = True
        E_prev = M_Z/2

        while t < tmax:
            dt = -np.log(np.random.random_sample()) / (4 * N_c * virtualtot)
            t += dt

            virtsum = np.cumsum(virtuals) / virtualtot
            random_dip = np.random.random_sample()
            position = len([v for v in virtsum if v < random_dip])

            newDip = GenRandV(event[position], event[position + 1])
            n3 = newDip
            q1, q2 = event[position], event[position + 1]

            if recoil_on:
                # mapping of recoil functions to args
                recoil_function_map = {
                    recoil.asymmetric_recoil: (q1, q2, n3, dt),
                    recoil.symmetric_recoil: (q1, q2, n3, dt),
                    recoil.panglobal_shower_dipole: (q1, q2, n3, event, dt),
                }
                args = recoil_function_map[recoil_function]
                if recoil_function is recoil.panglobal_shower_dipole:
                    q3, event = recoil_function(*args)
                else:
                    q1_new, q2_new, q3 = recoil_function(*args)
                    event = event[:position] + [q1_new, q3, q2_new] + event[position + 2:]
                thrust_val, thrust_vec = athrust(
                    np.array([v.spatial_part() for v in event], dtype=np.float64)
                )
            else:
                energy_scale = np.exp(-IAS * dt)
                denominator = FV.four_dot(q1, n3) + FV.four_dot(q2, n3)
                q3 = energy_scale * FV.four_dot(q1, q2) / denominator * n3 / FV.energy(n3)
                event = event[:position] + [q1, q3, q2] + event[position + 2:]
                thrust_val = None
                thrust_vec = np.array([0.0, 0.0, 1.0])

            E = q3.energy()
            emissions += 1

            # record data for this emission
            records.append({
                'run_id': x,
                'emission_id': emissions,
                't': t,
                'w': w,
                'event': [v.to_array().tolist() for v in event],
                'thrust_val': thrust_val,
                'thrust_vec': thrust_vec,
            })

            # incremental update of virtual corrections
            del virtuals[position]
            v1 = virtual_correction(event[position], event[position + 1])
            v2 = virtual_correction(event[position + 1], event[position + 2])
            virtuals.insert(position, v1)
            virtuals.insert(position + 1, v2)
            newvirtualtot = sum(virtuals)

            # veto according to old/new
            factor = virtualtot / newvirtualtot
            w *= factor
            virtualtot = newvirtualtot

    # After all events, save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)

    return df

from recoil import asymmetric_recoil, symmetric_recoil, panglobal_shower_dipole

if __name__ == '__main__':
    base_event = [FV(M_Z/2, 0, 0, M_Z/2), FV(M_Z/2, 0, 0, -M_Z/2)]
    num_events = 10_000
    t_start = 0.0
    init_w = 1.0

    output_dir = "csv_files"

    recoil_cases = [
        ('no_recoil', None, False), 
        ('asymmetric', asymmetric_recoil, True),
        ('symmetric', symmetric_recoil, True),
        ('panglobal', panglobal_shower_dipole, True)
    ]

    angle_list = np.linspace(10, 90, 10)  # degrees
    E_goals = np.linspace(10, M_Z / 2 - 1.0, 5)  # GeV

    for name, func, recoil_flag in recoil_cases:
        gap_matrix = np.zeros((len(angle_list), len(E_goals)))

        for i, angle in enumerate(angle_list):
            for j, E_goal in enumerate(E_goals):
                t_max = event_analysis.convert_energy_to_t(E_goal - 1)
                t_min = event_analysis.convert_energy_to_t(E_goal + 1)

                outfile = os.path.join(output_dir, f"shower_{name}_a{int(angle)}_E{int(E_goal)}.csv")                
                print(f"→ Running: {name}, θ={angle:.1f}°, E_goal={E_goal:.1f} GeV")

                df = shower(
                    origEvent=base_event,
                    NeV=num_events,
                    tstart=t_start,
                    tmax = t_max,
                    initialweight=init_w,
                    recoil_on=recoil_flag,
                    output_file=outfile,
                    angle_in_degrees=angle,
                    recoil_function=func
                )

                data = event_analysis.load_shower_data([outfile])
                subset = event_analysis.filter_by_t_range(data, t_min, t_max)
                gf = event_analysis.gap_fraction(
                    subset,
                    angle_in_degrees=angle,
                    t_min=t_min,
                    t_max=t_max
                )
                gap_matrix[i, j] = gf
                print(f"    Gap Fraction: {gf:.3f}")

        # Plot heatmap of gap fractions
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            gap_matrix,
            xticklabels=[f"{e:.0f}" for e in E_goals],
            yticklabels=[f"{a:.0f}°" for a in angle_list],
            annot=True, fmt=".2f", cmap="viridis", ax=ax
        )
        ax.set_xlabel("E_goal [GeV]")
        ax.set_ylabel("Angle [deg]")
        #ax.set_title(f"Gap Fraction Heatmap ({name})")
        plt.tight_layout()
        plt.savefig(f"gap_fraction_heatmap_energy_{name}.png")
        plt.close()
        print(f"Heatmap saved as 'gap_fraction_heatmap_energy_{name}.png'")