import numpy as np
import os
import time
import logging
from tqdm import tqdm
from four_vector import FourVector as FV
import event_analysis
import shower
from plotting import Hist
import plotting
import recoil
from typing import Callable

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHOWER_DIR = os.path.join(BASE_DIR, "shower_output")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis_output")

M_Z = 91.2
t_max = 0.1

def run_shower_test_angles(NeV: int, cone_angles: list):
    """run the shower for different recoil schemes, different cone angles and generate some plots

    Args:
        NeV (int): Number of events
        cone_angles (list): List of cone angles to use in the shower
    """
    from recoil import asymmetric_recoil, symmetric_recoil, panglobal_shower_dipole
    orig_event = [FV(M_Z/2, 0, 0, M_Z/2), FV(M_Z/2, 0, 0, -M_Z/2)]
    output_file = "shower_output.txt"

    recoil_schemes = [
        (None, "No Recoil"),
        (asymmetric_recoil, "Asymmetric"),
        (symmetric_recoil, "Symmetric"),
        (panglobal_shower_dipole, "PanGlobal")
    ]

    # Maps to store hists for each recoil scheme
    combined_S_E = { name: [] for _, name in recoil_schemes }
    combined_S_t = { name: [] for _, name in recoil_schemes }
    combined_E_out = { name: [] for _, name in recoil_schemes }

    for angle in cone_angles:
        angle_dir = os.path.join("histograms_different_angles", f"cone_{int(angle)}deg")
        os.makedirs(angle_dir, exist_ok=True)

        print(f"\n=== Running showers for cone angle: {angle} degrees ===\n")

        for recoil_func, recoil_name in recoil_schemes:
            baseline = recoil_func is None
            baseline_S_E, baseline_S_t, hist_E_outside, hist_t_outside, _ = shower.shower(
                origEvent=orig_event,
                NeV=NeV,
                tstart=0.0,
                tmax=0.1,
                initialweight=1.0,
                recoil_on=not baseline,
                output_file=output_file,
                angle_in_degrees=angle,
                recoil_function=recoil_func
            )

            combined_S_E[recoil_name].append(baseline_S_E)
            combined_S_t[recoil_name].append(baseline_S_t)
            combined_E_out[recoil_name].append(hist_E_outside)

    for recoil_name in combined_S_E.keys():
        Hist.plot_histograms_energy(
            combined_S_E[recoil_name],
            filename=f"histograms2/combined_S_E_{recoil_name}.pdf",
            xlabel=r"$E/E_{\max}$",
            NeV=NeV,
            ylabel=r"$S(E)$",
            title=f"S(E) for all cone angles [{recoil_name}]",
            cumulative=[True]*len(cone_angles),
            survival=[True]*len(cone_angles),
            labels=[f"{angle}°" for angle in cone_angles],
            normalize_energy_axis=True, 
            ylog=True
        )

        Hist.plot_histograms(
            combined_S_t[recoil_name],
            filename=f"histograms2/combined_S_t_{recoil_name}.pdf",
            xlabel="t",
            NeV=NeV,
            ylabel="S(t)",
            title=f"S(t) for all cone angles [{recoil_name}]",
            cumulative=[False]*len(cone_angles),
            survival=[False]*len(cone_angles),
            labels=[f"{angle}°" for angle in cone_angles]
        )

        Hist.plot_histograms_energy(
            combined_E_out[recoil_name],
            filename=f"histograms2/combined_E_outside_{recoil_name}.pdf",
            xlabel=r"$E_{\text{out}} / E_{\max}$",
            NeV=NeV,
            ylabel=r"$P(E_{\text{out}})$",
            title=f"P(E_out) for all cone angles [{recoil_name}]",
            cumulative=[False]*len(cone_angles),
            survival=[False]*len(cone_angles),
            labels=[f"{angle}°" for angle in cone_angles],
            normalize_energy_axis=True, 
            ylog = True
        )


plotting.clear_directory("histograms")
NeV = 10_000
cone_angles = [5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0]
run_shower_test_angles(NeV, cone_angles)
