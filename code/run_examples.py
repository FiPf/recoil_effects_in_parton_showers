__author__ = 'Fiona C. Pärli'
__email__ = 'fiona.paerli@students.unibe.ch'
__date__ = 'June 2025'

import os
from four_vector import FourVector as FV
import shower
import numpy as np
from plotting import Hist, clear_directory
from recoil import asymmetric_recoil, symmetric_recoil, panglobal_shower_dipole

M_Z = 91.2
t_max = 0.1

def run_shower_test(NeV: int):
    """run the shower for different recoil schemes

    Args:
        NeV (int): Number of events
    """
    orig_event = [FV(M_Z/2, 0, 0, M_Z/2), FV(M_Z/2, 0, 0, -M_Z/2)]
    output_file = "shower_output.txt"

    baseline_S_E, _, baseline_E_out, _, _ = shower.shower(
        origEvent=orig_event, NeV=NeV, tstart=0.0, tmax=t_max,
        initialweight=1.0, recoil_on=False, output_file=output_file,
        angle_in_degrees=60.0, recoil_function=None
    )

    recoil_funcs = [asymmetric_recoil, symmetric_recoil, panglobal_shower_dipole]
    S_E_all = [baseline_S_E]
    E_out_all = [baseline_E_out]
    labels = ["No Recoil"]

    for recoil_func in recoil_funcs:
        S_E, _, E_out, _, _ = shower.shower(
            origEvent=orig_event, NeV=NeV, tstart=0.0, tmax=t_max,
            initialweight=1.0, recoil_on=True, output_file=output_file,
            angle_in_degrees=60.0, recoil_function=recoil_func
        )
        S_E_all.append(S_E)
        E_out_all.append(E_out)
        labels.append(recoil_func.__name__)

    S_E_all[0].plot_histograms_energy(
        S_E_all,
        filename="histograms/combined_S_E_all.pdf",
        xlabel=r"$E/E_{\max}$", ylabel=r"$S(E)$", NeV=NeV,
        title="Sudakov $S(E)$ for Different Recoil Schemes",
        cumulative=[True] * len(S_E_all), survival=[True] * len(S_E_all),
        labels=labels, normalize_energy_axis=True
    )

    E_out_all[0].plot_histograms_energy(
        E_out_all,
        filename="histograms/combined_E_out_all.pdf",
        xlabel=r"$E_{\text{out}} / E_{\max}$", ylabel=r"$P(E_{\text{out}})$",
        NeV=NeV,
        title="Out-of-Cone $E_{\text{out}}$ for Different Recoil Schemes",
        cumulative=[False] * len(E_out_all), survival=[False] * len(E_out_all),
        labels=labels, normalize_energy_axis=True, ylog=True
    )


def run_shower_S_t_different_cuts(NeV: int):
    """run the shower for different rapidity cutoffs. Delete the "cut = 4" on top of the shower.py file.

    Args:
        NeV (int): Number of events
    """
    orig_event = [FV(M_Z/2, 0, 0, M_Z/2), FV(M_Z/2, 0, 0, -M_Z/2)]
    output_file = "shower_output.txt"
    cuts = [2, 3, 4, 5]

    S_t_all = []
    labels = []
    for cut in cuts:
        shower.cut = cut
        print(f"Running for cut = {cut}")
        _, bin_S_of_t, _, _, _ = shower.shower(
            origEvent=orig_event, NeV=NeV, tstart=0.0, tmax=t_max,
            initialweight=1.0, recoil_on=True, output_file=output_file,
            angle_in_degrees=60.0, recoil_function=panglobal_shower_dipole
        )
        S_t_all.append(bin_S_of_t)
        labels.append(r"$\eta_{\text{cut}}$ = " + str(cut))

    S_t_all[0].plot_histograms(
        S_t_all,
        filename="histograms/combined_S_t_cuts_PanGlobal.pdf",
        xlabel="t", ylabel="S(t)", NeV=NeV,
        title="Sudakov $S(t)$ for Different Rapidity Cuts",
        cumulative=[False] * len(S_t_all), survival=[False] * len(S_t_all),
        labels=labels
    )


def run_shower_test_angles(NeV: int, cone_angles: list):
    """run the shower for different recoil schemes, different cone angles and generate some plots

    Args:
        NeV (int): Number of events
        cone_angles (list): List of cone angles to use in the shower
    """
    orig_event = [FV(M_Z/2, 0, 0, M_Z/2), FV(M_Z/2, 0, 0, -M_Z/2)]
    output_file = "shower_output.txt"

    recoil_schemes = [
        (None, "No Recoil"),
        (asymmetric_recoil, "Asymmetric"),
        (symmetric_recoil, "Symmetric"),
        (panglobal_shower_dipole, "PanGlobal")
    ]

    combined_S_E = { name: [] for _, name in recoil_schemes }
    combined_S_t = { name: [] for _, name in recoil_schemes }
    combined_E_out = { name: [] for _, name in recoil_schemes }

    for angle in cone_angles:
        print(f"\n=== Running showers for cone angle: {angle} degrees ===")
        for recoil_func, recoil_name in recoil_schemes:
            S_E, S_t, E_out, _, _ = shower.shower(
                origEvent=orig_event, NeV=NeV, tstart=0.0, tmax=t_max,
                initialweight=1.0, recoil_on=(recoil_func is not None),
                output_file=output_file, angle_in_degrees=angle,
                recoil_function=recoil_func
            )
            combined_S_E[recoil_name].append(S_E)
            combined_S_t[recoil_name].append(S_t)
            combined_E_out[recoil_name].append(E_out)

    for recoil_name in combined_S_E:
        Hist.plot_histograms_energy(
            combined_S_E[recoil_name],
            filename=f"histograms/S_E_{recoil_name}.pdf",
            xlabel=r"$E/E_{\max}$", ylabel=r"$S(E)$", NeV=NeV,
            title=f"Sudakov $S(E)$ vs Cone Angle [{recoil_name}]",
            cumulative=[True]*len(cone_angles), survival=[True]*len(cone_angles),
            labels=[f"{angle}°" for angle in cone_angles],
            normalize_energy_axis=True, ylog=True
        )

        Hist.plot_histograms(
            combined_S_t[recoil_name],
            filename=f"histograms/S_t_{recoil_name}.pdf",
            xlabel="t", ylabel="S(t)", NeV=NeV,
            title=f"Sudakov $S(t)$ vs Cone Angle [{recoil_name}]",
            cumulative=[False]*len(cone_angles), survival=[False]*len(cone_angles),
            labels=[f"{angle}°" for angle in cone_angles]
        )

        Hist.plot_histograms_energy(
            combined_E_out[recoil_name],
            filename=f"histograms/E_out_{recoil_name}.pdf",
            xlabel=r"$E_{\text{out}} / E_{\max}$", ylabel=r"$P(E_{\text{out}})$",
            NeV=NeV,
            title=f"Out-of-Cone Energy vs Cone Angle [{recoil_name}]",
            cumulative=[False]*len(cone_angles), survival=[False]*len(cone_angles),
            labels=[f"{angle}°" for angle in cone_angles],
            normalize_energy_axis=True, ylog=True
        )

def run_shower_soft_limit_scan(NeV: int):
    """Run the shower for different IAS values to test the soft limit scaling.

    Args:
        NeV (int): Number of events
    """
    orig_event = [FV(M_Z/2, 0, 0, M_Z/2), FV(M_Z/2, 0, 0, -M_Z/2)]
    angle = 60.0

    ias_values = [1/0.001, 1/0.01, 1/0.1]
    alpha_s_values = [4 * np.pi / ias for ias in ias_values]
    labels = [f"α_s={als:.3f}" for als in alpha_s_values]

    standard_ias = 106.495
    standard_alpha_s = 4 * np.pi / standard_ias
    standard_label = f"NoRecoil α_s={standard_alpha_s:.3f}"

    shower.IAS = standard_ias
    print(f"Computing baseline NoRecoil at α_s={standard_alpha_s:.3f}...")
    S_E_baseline, S_t_baseline, _, _, _ = shower.shower(
        origEvent=orig_event, NeV=NeV, tstart=0.0, tmax=t_max,
        initialweight=1.0, recoil_on=False, output_file=None,
        angle_in_degrees=angle, recoil_function=None, IAS=standard_ias
    )

    recoil_schemes = [
        (None, "No Recoil"),
        (asymmetric_recoil, "Asymmetric"),
        (symmetric_recoil, "Symmetric"),
        (panglobal_shower_dipole, "PanGlobal")
    ]

    combined_S_E = {name: [] for _, name in recoil_schemes}
    combined_S_t = {name: [] for _, name in recoil_schemes}

    for func, name in recoil_schemes:
        for ias, als in zip(ias_values, alpha_s_values):
            shower.IAS = ias
            print(f"  α_s = {als:.3f} [{name}]")
            S_E, S_t, _, _, _ = shower.shower(
                origEvent=orig_event, NeV=NeV, tstart=0.0, tmax=t_max,
                initialweight=1.0, recoil_on=(func is not None),
                output_file=None, angle_in_degrees=angle,
                recoil_function=func, IAS=ias
            )
            combined_S_E[name].append(S_E)
            combined_S_t[name].append(S_t)

    for name in combined_S_E:
        S_E_list = combined_S_E[name].copy()
        S_t_list = combined_S_t[name].copy()
        plot_labels = labels.copy()
        if name != "No Recoil":
            S_E_list.insert(0, S_E_baseline)
            S_t_list.insert(0, S_t_baseline)
            plot_labels.insert(0, standard_label)

        Hist.plot_histograms_energy(
            S_E_list,
            filename=f"histograms/combined_S_E_IAS_{name}.pdf",
            xlabel=r"$E/E_{\max}$", ylabel=r"$S(E)$", NeV=NeV,
            title=f"Sudakov $S(E)$ for Different α_s [{name}]",
            cumulative=[True] * len(plot_labels),
            survival=[True] * len(plot_labels),
            labels=plot_labels, normalize_energy_axis=True, ylog=True
        )

        Hist.plot_histograms(
            S_t_list,
            filename=f"histograms/combined_S_t_IAS_{name}.pdf",
            xlabel="t", ylabel="S(t)", NeV=NeV,
            title=f"Sudakov $S(t)$ for Different α_s [{name}]",
            cumulative=[False] * len(plot_labels),
            survival=[False] * len(plot_labels),
            labels=plot_labels, ylog=True
        )

if __name__ == "__main__":
    clear_directory("histograms")
    NeV = 10_000

    run_shower_test(NeV)
    run_shower_S_t_different_cuts(NeV)
    run_shower_test_angles(NeV, [5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0])
    run_shower_soft_limit_scan(NeV)