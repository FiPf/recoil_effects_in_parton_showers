import numpy as np
import os
from four_vector import FourVector as FV
import shower
from plotting import Hist
import plotting
from recoil import asymmetric_recoil, symmetric_recoil, panglobal_shower_dipole

M_Z = 91.2
t_max = 0.1

def run_shower_S_t_different_cuts(NeV: int):
    """run the shower for different rapidity cutoffs. Delete the "cut = 4" on top of the shower.py file.

    Args:
        NeV (int): Number of events
    """
    orig_event = [FV(M_Z/2, 0, 0, M_Z/2), FV(M_Z/2, 0, 0, -M_Z/2)]
    output_file = "shower_output.txt"

    cuts = [2, 3, 4, 5]

    for scheme in [panglobal_shower_dipole]: 
        S_t_all = []
        labels = []
        for my_cut in cuts:
            cut = my_cut  # set the global cut used in virtual_correction
            print(f"Running for cut = {cut}")
            shower.cut = my_cut
            bin_S_of_E, bin_S_of_t, _, _, _ = shower.shower(
                origEvent=orig_event,
                NeV=NeV,
                tstart=0.0,
                tmax=t_max,
                initialweight=1.0,
                recoil_on=True,
                output_file=output_file,
                angle_in_degrees=60.0,
                recoil_function=scheme
            )
            S_t_all.append(bin_S_of_t)
            labels.append(r"$\eta_{cut}$ = " + f"{cut}")

        # Plot combined S(t)
        S_t_all[0].plot_histograms(
            S_t_all,
            filename=os.path.join("histograms2", f"combined_S_t_cuts_{scheme.__name__}.pdf"),
            xlabel="t",
            ylabel="S(t)",
            NeV=NeV,
            title="Sudakov $S(t)$ for Different Cuts",
            cumulative=[False] * len(S_t_all),
            survival=[False] * len(S_t_all),
            labels=labels,
            normalize_energy_axis=False
        )

if __name__ == "__main__":
    plotting.clear_directory("histograms")
    NeV = 5_000
    run_shower_S_t_different_cuts(NeV)
