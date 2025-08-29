# Recoil Effects in Parton Showers
Code for my Bachelor's Thesis

## Remarks on the Implementation of the Parton Shower

This section contains a brief overview of the parton shower implementation in Python and Rust. A cleaned-up version of the code, which enables the reader to reproduce the results presented in this bachelor's thesis, can be found at the following link:  
[https://github.com/FiPf/recoil_effects_in_parton_showers.git](https://github.com/FiPf/recoil_effects_in_parton_showers.git)

### Structure of the Code

The GitHub repository contains three essential folders, whose contents I will list and briefly explain. Everything else should be self-explanatory.

- **Folder 1: Code**  
  This folder contains the main implementation of the parton shower and its analysis tools:
  - **shower.py:** Implements the parton shower algorithm, described in *Section `Algorithm`*.
  - **shower_energy_gap_fraction.py:** Contains the same parton shower algorithm, but stores the events in a dataframe, which allows for computing and plotting the gap fraction heatmap.
  - **running_coupling_shower.py:** Contains the parton shower algorithm, but adapted for the running coupling instead of the fixed coupling at $\alpha_s(Q)$ for the scale $Q$.
  - **run_examples.py:** Provides functions to run typical shower configurations, test different recoil schemes, rapidity cutoffs, cone angles, and generate the plots shown in this thesis.
  - **recoil.py:** Defines the different recoil schemes, namely asymmetric recoil, symmetric recoil, and the PanGlobal recoil schemes.
  - **four_vector.py:** Defines the `FourVector` class with Lorentz vector operations, dot products, boosts, and basic transformations. The implementation of this class is based on the code at [https://github.com/MarcelBalsiger/ngl_resum](https://github.com/MarcelBalsiger/ngl_resum), which is associated with this paper: [Balsiger et al. 2020](https://doi.org/10.1007/JHEP09(2020)029).
  - **plotting.py:** Contains functions for histogramming, plotting Sudakov form factors, energy distributions, and clearing output directories.
  - **event_analysis.py:** Provides analysis routines and helper functions for computing quantities such as out-of-cone energy and gap fractions from shower events.
  - **transform.py:** Contains transformations, such as Lorentz transformations, Householder transformations, and rotations, used internally by the shower and recoil schemes.

- **Folder 2: Exercises**  
  This folder contains a simple toy model and Monte Carlo integration examples:
  - **main.ipynb:** Jupyter notebook with tests for Monte Carlo integration and code used for the toy model discussed in *Section `Toy Model`*.
  - **monte_carlo.py:** Basic Monte Carlo integrator illustrating the method, along with tests and functions for the analytical solutions to the toy model.

- **Folder 3: thrust_acceleration_code**  
  This folder contains the Rust implementation of the thrust axis and thrust value computation, described in *Section `Thrust Algorithm`*. The Rust version significantly improves performance compared to a pure Python implementation.

The code is structured to be modular and hopefully easy to extend or adapt for future studies.

## Important Note
If you are unable to use the Rust thrust function, you can use the Python code. In the file where the shower is stored (for example, shower.py), uncomment the import of the Python version at the top of the file and comment out the Rust version. However, this will signficantly slow down the process. 
