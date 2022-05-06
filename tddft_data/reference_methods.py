import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def angle_error(vec1, vec2):
    numerator = np.dot(vec1, vec2)
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    return np.rad2deg(np.arccos(numerator/denominator))

def angle_error_with_phase(vec1, vec2):
    result = min(angle_error(vec1, vec2), angle_error(-vec1, vec2))
    return result

def get_Na_Nc_vec(xyz_file):
    coords = list(open(xyz_file))
    
    Na_line = 18
    Nc_line = 35

    Na = np.array([float(x) for x in re.findall(r'-?\d+\.\d+', coords[Na_line])])
    Nc = np.array([float(x) for x in re.findall(r'-?\d+\.\d+', coords[Nc_line])])

    return Na-Nc

def read_gaussian_result(chromophore_name, method, root):

    file_name = f"{root}/{chromophore_name}/{method}_{chromophore_name}.log"
        
    last_transition_energy = 0.
    last_index = None

    homo_lumo_energy = 0.
    homo_lumo_coeff = 0.
    homo_lumo_index = None

    transition_dipoles = []

    lines = list(open(file_name))
    for enum, line in enumerate(lines):
        if "Ground to excited state transition electric dipole moments (Au):" in line:
            for i in range(5):
                transition_dipole = np.array([float(x) for x in re.findall(r'-?\d+.\d+', lines[enum+2+i])[:3]])
                
                transition_dipoles.append(transition_dipole)
        
        
        if "Excited State" in line:
            last_index = int(re.findall(r'\d+', line)[0])
            last_transition_energy = float(re.findall(r'\d+.\d+', line)[0])

        if "->" in line:
            orbitals = [int(x) for x in re.findall(r'\d+', line)[:2]]
            coeff = abs(float(re.findall(r'-?\d+.\d+', line)[2]))

            if orbitals[1] - orbitals[0] == 1 and coeff > homo_lumo_coeff:
                homo_lumo_energy = last_transition_energy
                homo_lumo_index = last_index
                homo_lumo_coeff = coeff
               
    Qy_dipole = transition_dipoles[homo_lumo_index-1]

    Na_Nc_vec = get_Na_Nc_vec(f"{root}/{chromophore_name}/{chromophore_name}.xyz")
    angle_error = angle_error_with_phase(Na_Nc_vec, Qy_dipole)

    return homo_lumo_energy, Qy_dipole, angle_error

def read_gaussian_method(method, root="."):    
    results = {
        "chromophore" : [],
        f"{method} excitation energy (hr)" : [],
        f"{method} excitation energy (eV)" : [],
        f"{method} transition dipole" : [],
        f"{method} transition dipole magnitude" : [],
        f"{method} angle to Na_Nc" : []
    }
    
    steps = range(1, 1900, 50)
    chromophores = range(1, 28, 1)
    
    for step in steps:
        for chromophore in chromophores:
            chromophore_name = f"step_{step}_chromophore_{chromophore}"

            homo_lumo_energy, transition_dipole, angle_error = read_gaussian_result(chromophore_name, method, root)

            results["chromophore"].append(chromophore_name)
            results[f"{method} excitation energy (hr)"].append(homo_lumo_energy / 27.2114)
            results[f"{method} excitation energy (eV)"].append(homo_lumo_energy)
            results[f"{method} transition dipole"].append(transition_dipole)
            results[f"{method} transition dipole magnitude"].append(np.linalg.norm(transition_dipole))
            results[f"{method} angle to Na_Nc"].append(angle_error)
            
    df_result = pd.DataFrame.from_dict(results)
    return df_result.set_index("chromophore")

def read_td_result(chromophore_name, method, root):

    file_name = f"{root}/{chromophore_name}/{method}_{chromophore_name}.out"
    
    if method == "PBE0":
        file_name = f"{root}/{chromophore_name}/{chromophore_name}.out"
        
    last_transition_energy = 0.
    last_index = None

    homo_lumo_energy = 0.
    homo_lumo_coeff = 0.
    homo_lumo_index = None

    transition_dipole = None

    lines = list(open(file_name))
    for enum, line in enumerate(lines):
        if "Excited State" in line:
            last_index = int(re.findall(r'\d+', line)[0])
            last_transition_energy = float(re.findall(r'\d+.\d+', line)[0])

        if "->" in line:
            orbitals = [int(x) for x in re.findall(r'\d+', line)[:2]]
            coeff = abs(float(re.findall(r'-?\d+.\d+', line)[2]))

            if orbitals[1] - orbitals[0] == 1 and coeff > homo_lumo_coeff:
                homo_lumo_energy = last_transition_energy
                homo_lumo_index = last_index
                homo_lumo_coeff = coeff
        
        if "Transition Dipole Moments (a.u.)" in line:
            transition_dipole = np.array([float(x) for x in re.findall(r'-?\d+.\d+', lines[enum+1+homo_lumo_index])])

    Na_Nc_vec = get_Na_Nc_vec(f"{root}/{chromophore_name}/{chromophore_name}.xyz")
    angle_error = angle_error_with_phase(Na_Nc_vec, transition_dipole)

    return homo_lumo_energy, transition_dipole, angle_error

def read_td_method(method, root="."):    
    results = {
        "chromophore" : [],
        f"{method} excitation energy (hr)" : [],
        f"{method} excitation energy (eV)" : [],
        f"{method} transition dipole" : [],
        f"{method} transition dipole magnitude" : [],
        f"{method} angle to Na_Nc" : []
    }
    
    steps = range(1, 1900, 50)
    chromophores = range(1, 28, 1)
    
    for step in steps:
        for chromophore in chromophores:
            chromophore_name = f"step_{step}_chromophore_{chromophore}"

            try:
                homo_lumo_energy, transition_dipole, angle_error = read_td_result(chromophore_name, method, root)
        
                results["chromophore"].append(chromophore_name)
                results[f"{method} excitation energy (hr)"].append(homo_lumo_energy)
                results[f"{method} excitation energy (eV)"].append(homo_lumo_energy * 27.2114)
                results[f"{method} transition dipole"].append(transition_dipole)
                results[f"{method} transition dipole magnitude"].append(np.linalg.norm(transition_dipole))
                results[f"{method} angle to Na_Nc"].append(angle_error)
            except:
                continue
            
    df_result = pd.DataFrame.from_dict(results)
    return df_result.set_index("chromophore")


def read_excited_scf_result(chromophore_name, method, root):
    file_name = f"{root}/{chromophore_name}/{method}_{chromophore_name}.out"

    lines = list(open(f'{file_name}'))
            
    energy_hr = None
    transition_dipole = None
            
    for line in lines:
        if "Excitation energy:" in line:
            energy_hr = float(re.findall(r'\d+\.\d+', line)[0])
                    
        if "Transition dipole" in line:
            transition_dipole = np.array([float(x) for x in re.findall(r'-?\d+\.\d+', line)])
                        
    if energy_hr is None or transition_dipole is None:
        return None, None, None
                
    Na_Nc_vec = get_Na_Nc_vec(f'{root}/{chromophore_name}/{chromophore_name}.xyz')
    magnitude = np.linalg.norm(transition_dipole)
    angle_error = angle_error_with_phase(Na_Nc_vec, transition_dipole)
    return energy_hr, transition_dipole, angle_error 
    
def read_excited_scf_method(method, root="."):    
    results = {
        "chromophore" : [],
        f"{method} excitation energy (hr)" : [],
        f"{method} excitation energy (eV)" : [],
        f"{method} transition dipole" : [],
        f"{method} transition dipole magnitude" : [],
        f"{method} angle to Na_Nc" : []
    }
    
    steps = range(1, 1900, 50)
    chromophores = range(1, 28, 1)
    
    for step in steps:
        for chromophore in chromophores:
            chromophore_name = f"step_{step}_chromophore_{chromophore}"

            energy, transition_dipole, angle_error = read_excited_scf_result(chromophore_name, method, root)
    
            if energy is None or transition_dipole is None:
                continue

            results["chromophore"].append(chromophore_name)
            results[f"{method} excitation energy (hr)"].append(energy)
            results[f"{method} excitation energy (eV)"].append(energy * 27.2114)
            results[f"{method} transition dipole"].append(transition_dipole)
            results[f"{method} transition dipole magnitude"].append(np.linalg.norm(transition_dipole))
            results[f"{method} angle to Na_Nc"].append(angle_error)                
           
    df_result = pd.DataFrame.from_dict(results)
    return df_result.set_index("chromophore")

def merge_multiple(dfs, so_far=None):
    if so_far is None:
        so_far = pd.merge(dfs[-1], dfs[-2], left_index=True, right_index=True)
        return merge_multiple(dfs[:-2], so_far)
    
    if len(dfs) == 1:
        so_far = pd.merge(dfs[0], so_far, left_index=True, right_index=True)
        return so_far
    
    if len(dfs) == 0:
        return so_far        
    
    if len(dfs) > 1:
        so_far = pd.merge(dfs[-1], so_far, left_index=True, right_index=True)
        return merge_multiple(dfs[:-1], so_far)