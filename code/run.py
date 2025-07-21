import numpy as np
import os
from four_vector import FourVector as FV
import shower
from plotting import Hist
import plotting
import recoil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHOWER_DIR = os.path.join(BASE_DIR, "shower_output")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis_output")

M_Z = 91.2
t_max = 0.1

def run_shower_test(NeV: int):
    """run the shower for different recoil schemes

    Args:
        NeV (int): Number of events
    """
    from recoil import asymmetric_recoil, symmetric_recoil, panglobal_shower_dipole

    orig_event = [FV(M_Z/2, 0, 0, M_Z/2), FV(M_Z/2, 0, 0, -M_Z/2)]
    output_file = "shower_output.txt"

    # 1) Run no-recoil baseline
    baseline_S_E, baseline_S_t, baseline_E_out, baseline_t_out, baseline_E_out_x3bins = shower.shower(
        origEvent=orig_event,
        NeV=NeV,
        tstart=0.0,
        tmax=0.1,
        initialweight=1.0,
        recoil_on=False,
        output_file=output_file,
        angle_in_degrees=60.0,
        recoil_function=None
    )

    # 2) Run each recoil scheme and store results
    recoil_funcs = [asymmetric_recoil, symmetric_recoil, panglobal_shower_dipole]
    S_E_all = [baseline_S_E]
    E_out_all = [baseline_E_out]
    labels = ["No Recoil"]

    for recoil_func in recoil_funcs:
        S_E, _, E_out, _, _ = shower.shower(
            origEvent=orig_event,
            NeV=NeV,
            tstart=0.0,
            tmax=0.1,
            initialweight=1.0,
            recoil_on=True,
            output_file=output_file,
            angle_in_degrees=60.0,
            recoil_function=recoil_func
        )
        S_E_all.append(S_E)
        E_out_all.append(E_out)
        labels.append(recoil_func.__name__)

    # 3) Combined S(E) plot
    S_E_all[0].plot_histograms_energy(
        S_E_all,
        filename=os.path.join("histograms2", "combined_S_E_all.pdf"),
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
        filename=os.path.join("histograms2", "combined_E_out_all.pdf"),
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
    plotting.clear_directory("histograms2")
    NeV = 10_000
    run_shower_test(NeV)
