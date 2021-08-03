import os
import json
import subprocess

import numpy as np


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def calc_dipole_error(vec1, vec2):
    """
    find the error between two vectors as the norm of the difference.
    Altered due to arbitary phase of transition dipoles

    >>> calc_dipole_error([0,3,3], [0,1,1])
    2.8284271247461903

    >>> calc_dipole_error([0,3,3], [0,-1,-1])
    2.8284271247461903


    """
    phase1 = np.linalg.norm(np.array(vec2) - np.array(vec1))
    phase2 = np.linalg.norm(np.array(vec2) + np.array(vec1))

    return min(phase1, phase2)


def calc_angle_error(vector_1, vector_2):
    """
	computes the angle between two transition dipoles.
	Altered due to arbitary phase of transition dipoles

	>>> calc_angle_error([0,0,1], [0,1,1])
	45.00000000000001

	>>> calc_angle_error([0,0,1], [0,-1,-1])
	45.0
	"""
    if (np.linalg.norm(vector_1) < 1e-6 and np.linalg.norm(vector_2) < 1e-6):
        return 0.0

    elif (np.linalg.norm(vector_1) < 1e-6 or np.linalg.norm(vector_2) < 1e-6):
        return 90

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    angle = angle if angle < (np.pi / 2) else np.pi - angle

    return np.rad2deg(angle)


def make_ref_data(file_name):
    """
    make dictionary object of the TD-DFT reference data

    >>> data = make_ref_data("tddft_results.json")
    >>> data["step_1_chromophore_01"]["energy"]
    0.067717
    >>> data["step_1_chromophore_01"]["transition_dipole"]
    [0.305422706, -2.699393089, 0.410750669]

    >>> type(data)
    <class 'dict'>

    """
    data = open(file_name)
    data = json.load(data)

    return data


def run_qcore(input_tuple):
    """
    runs qcore with the input string
    """

    chromophore, chromophore_str = input_tuple

    qcore_path = os.environ["QCORE_PATH"]
    #qcore_path = "~/.local/src/Qcore/release/qcore"
    # qcore_path = "~/qcore/cmake-build-release/bin/qcore"
    json_str = " -n 1 -f json --schema none -s "
    norm_str = " -n 1 -s "

    try:
        json_run = subprocess.run(qcore_path + json_str + chromophore_str,
                                  shell=True,
                                  stdout=subprocess.PIPE,
                                  executable="/bin/bash",
                                  universal_newlines=True)

        json_results = json.loads(json_run.stdout)

        return [chromophore, json_results, True]

    except:

        return [chromophore, None, False]


def make_output_func(file_name):
    if file_name.endswith(".out"):
        file = open(file_name, 'w')
        return lambda x: print(x, file=file)
    else:
        file = open(file_name + ".out", 'w')
        return lambda x: print(x, file=file)
