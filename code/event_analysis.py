__author__ = 'Fiona C. Pärli'
__email__ = 'fiona.paerli@students.unibe.ch'
__date__ = 'June 2025'

import numpy as np
from four_vector import FourVector as FV
import os
from plotting import Hist
import time
from tqdm import tqdm
import plotting
from athrust_accel import athrust  # Import the Rust-accelerated function
import pandas as pd
import glob
import ast
from typing import List, Union, Callable
import re
import matplotlib.pyplot as plt

M_Z = 91.2
out_dir = "analysis_output"
os.makedirs(out_dir, exist_ok=True)

def energy_out(fourVector: FV, thrustVector: np.ndarray, angle_in_degrees: float):
    """compute the energy that is outside the cone for a given cone opening angle in degrees

    Args:
        fourVector (FV): FourVector to find out whether it is outside the cone or not
        thrustVector (np.ndarray): thrust vector of the event
        angle_in_degrees (float): cone opening angle in degrees

    Returns:
        float: energy that is outside the cone
    """
    three_vec = FV.spatial_part(fourVector)
    norm = np.linalg.norm(three_vec)
    direction = three_vec / norm
    costheta = np.dot(direction, thrustVector)
    energy = 0
    if abs(costheta) < np.cos(np.radians(angle_in_degrees)):
        energy = FV.energy(fourVector)
    return energy

def convert_energy_to_t(E: float, Q_0: float = M_Z/2): 
    """formula to convert energy to evolution time variable t

    Args:
        E (float): energy
        Q_0 (float, optional): Scale. Defaults to M_Z/2.

    Returns:
        float: t variable
    """
    IAS = 106.495
    t = np.log(E/Q_0)*(-1/IAS)
    return t

def filter_by_t_range(df: pd.DataFrame, t_min: float, t_max: float) -> pd.DataFrame:
    """filer out all events in the interval [t_min, t_max], so they can be processed later

    Args:
        df (pd.DataFrame): Dataframe containing all events
        t_min (float): lower boundary of the interval
        t_max (float): upper boundary of the interval

    Returns:
        pd.DataFrame: Dataframe with the events in the interval [t_min, t_max]
    """
    mask = (df['t'] >= t_min) & (df['t'] <= t_max)
    return df.loc[mask].reset_index(drop=True)

def gap_fraction(df: pd.DataFrame, angle_in_degrees: float, t_min: float, t_max: float) -> float:
    """compute the gap fraction for given target energy and cone opening angle

    Args:
        df (pd.DataFrame): Dataframe with the events in the interval [t_min, t_max]
        angle_in_degrees (float): cone opening angle
        t_min (float): lower boundary of the interval
        t_max (float): upper boundary of the interval

    Returns:
        float: gap fraction
    """
    total_weight = df['w'].sum()
    print(total_weight)
    if total_weight == 0 or t_max <= t_min:
        return 0.0

    survived_weight = 0.0
    for wgt, vecs, thrust in zip(df['w'], df['event'], df['thrust_vec']):
        thrust_np = np.array(thrust)
        has_out_of_cone = False
        for v in vecs:
            fv = FV(*v)
            if energy_out(fv, thrust_np, angle_in_degrees) != 0.0:
                has_out_of_cone = True
                break

        if not has_out_of_cone:
            survived_weight += wgt

    fraction = survived_weight / total_weight
    return fraction

def energy_weighted_gap_fraction(df: pd.DataFrame, angle_in_degrees: float, t_min: float, t_max: float) -> float:
    """compute the gap fraction for given target energy and cone opening angle, this time with energy weights

    Args:
        df (pd.DataFrame): Dataframe with the events in the interval [t_min, t_max]
        angle_in_degrees (float): cone opening angle
        t_min (float): lower boundary of the interval
        t_max (float): upper boundary of the interval

    Returns:
        float: gap fraction energy weighted
    """
    total_weight = df['w'].sum()
    print(total_weight)
    if total_weight == 0 or t_max <= t_min:
        return 0.0

    total_weighted_energy = 0.0
    survived_weighted_energy = 0.0
    for wgt, vecs, thrust in zip(df['w'], df['event'], df['thrust_vec']):

        E_i = sum(FV(*v).energy() for v in vecs)
        total_weighted_energy += wgt * E_i

        thrust_np = np.array(thrust)
        has_out_of_cone = False
        for v in vecs:
            fv = FV(*v)
            if energy_out(fv, thrust_np, angle_in_degrees) != 0.0:
                has_out_of_cone = True
                break

        if not has_out_of_cone:
            survived_weighted_energy += wgt * E_i #energy weights!

    if total_weighted_energy == 0.0:
        return 0.0

    return survived_weighted_energy / total_weighted_energy

def load_shower_data(files: Union[str, List[str]]) -> pd.DataFrame:
    """ Load one or more Monte Carlo shower output CSV files and parse them into a combined DataFrame. 
    This function reads CSV files produced by the `shower` routine, which contain
    emission records for each event. 

    Args:
        files (Union[str, List[str]]): file pattern or list of file paths to loas

    Returns:
        pd.DataFrame: A single combined DataFrame with parsed columns:
            - 'event': list of four-vectors (each as list of floats)
            - 'thrust_vec': thrust axis vector (list of floats)
            - 'w': event weight (float)
            - 't': evolution variable (float)
            - 'source_file': the original file this row came from
    """
    if isinstance(files, str):
        file_list = glob.glob(files)
    else:
        file_list = files

    df_list = []
    for f in file_list:
        tmp = pd.read_csv(f, dtype=str)

        def safe_eval_array(x):
            if isinstance(x, str) and 'nan' in x.lower():
                return None
            try:
                val = eval(x, {"np": np, "array": np.array})
                return [v.tolist() if isinstance(v, np.ndarray) else v for v in val]
            except Exception:
                return None
        
        def parse_thrust_vec(s):
            if not isinstance(s, str):
                return None
            try:
                return ast.literal_eval(s)
            except (SyntaxError, ValueError):
                try:
                    return np.fromstring(s.strip("[]"), sep=' ').tolist()
                except Exception:
                    return None

        tmp['event'] = tmp['event'].apply(safe_eval_array)
        tmp['thrust_vec'] = tmp['thrust_vec'].apply(parse_thrust_vec)
        tmp['w'] = tmp['w'].astype(float)
        tmp['t'] = tmp['t'].astype(float)
        tmp['source_file'] = f
        df_list.append(tmp)

    return pd.concat(df_list, ignore_index=True)
