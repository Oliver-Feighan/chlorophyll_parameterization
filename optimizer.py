import scipy.optimize
from scipy.optimize import minimize
from scipy.stats import linregress

import enum
import argparse
import subprocess
import random
import json
import numpy as np
import random
import time
import datetime
import sys
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt


CLI=argparse.ArgumentParser()
CLI.add_argument(
	"--params",
	nargs="*",
	type=str,
	default=[],
	help="active parameters for optimization",
)

CLI.add_argument(
	"--samples",
	nargs=1,
	type=int,
	default=100,
	help="the number of samples in the training set")

CLI.add_argument(
	"--weights",
	nargs=3,
	type=float,
	default=[1., 1., 1.],
	help="""the weights associated with the MAE of the energy, the R^2 coefficient,
 and the MAE of the transition dipole""")

CLI.add_argument(
	"--method",
	nargs=1,
	type=str,
	default=["Nelder-Mead"],
	choices=["Nelder-Mead", "test", "Bayesian_Gaussian_Process"],
	help="specify optimization method, or flag to run a validation set"
)

CLI.add_argument(
	"--max_iter",
	nargs=1,
	type=int,
	default=5000,
	help="maximum number of iterations for optimization method"
)

CLI.add_argument(
	"--ref_data",
	nargs=1,
	type=str,
	default='tddft_data/tddft_data.json',
	help="json file that stores reference data, used to optimize against"
)

CLI.add_argument(
	"--run_tests",
	nargs=1,
	type=bool,
	default=False,
	help="bool to flag running doctests instead of an optimization"
)

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
	if(np.linalg.norm(vector_1) < 1e-6 and np.linalg.norm(vector_2) < 1e-6):
		return 0.0

	elif(np.linalg.norm(vector_1) < 1e-6 or np.linalg.norm(vector_2) < 1e-6):
		return 90

	unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
	unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
	dot_product = np.dot(unit_vector_1, unit_vector_2)
	angle = np.arccos(dot_product)
	angle = angle if angle < (np.pi/2) else np.pi - angle

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


def run_qcore(chromophore_str):
	"""
	runs qcore with the input string
	"""
	qcore_path = "~/qcore/cmake-build-release/bin/qcore"
	#qcore_path = "~/.local/src/Qcore/release/qcore"
	json_str = " -n 1 -f json --schema none -s "
	norm_str = " -n 1 -s "

	json_run = subprocess.run(qcore_path + json_str + chromophore_str,
					shell=True,
					stdout=subprocess.PIPE,
					executable="/bin/bash",
					universal_newlines=True)

	if json_run.returncode != 0:
		norm_run = subprocess.run(qcore_path + norm_str + chromophore_str,
					shell=True,
					stdout=subprocess.PIPE,
					executable="/bin/bash",
					universal_newlines=True)

		print(norm_run.stdout)

		exit()

	return json.loads(json_run.stdout)

class Errors():
	def make_full_error_lists(self, results):
		for i in results.values():
			angle_error = calc_angle_error(i["tddft_dipole"], i["xtb_dipole"])
			if angle_error < 20:
				self.chromophores.append(i)
				self.Na_Ncs.append(i["Na_Nc"])
				self.tddft_energies.append(i["tddft_energy"])
				self.xtb_energies.append(i["xtb_energy"])
				self.tddft_dipoles.append(i["tddft_dipole"])
				self.xtb_dipoles.append(i["xtb_dipole"])
				self.tddft_dipole_mags.append(np.linalg.norm(i["tddft_dipole"]))
				self.xtb_dipole_mags.append(np.linalg.norm(i["xtb_dipole"]))

				self.energy_errors.append(i["xtb_energy"] - i["tddft_energy"])
				self.dipole_errors.append(calc_dipole_error(i["xtb_dipole"], i["tddft_dipole"]))
				self.tddft_angle_errors.append(calc_angle_error(i["Na_Nc"], i["tddft_dipole"]))
				self.xtb_angle_errors.append(calc_angle_error(i["Na_Nc"], i["xtb_dipole"]))

		self.tddft_energies 	= np.array(self.tddft_energies)
		self.xtb_energies 		= np.array(self.xtb_energies)
		self.energy_errors 		= np.array(self.energy_errors)
		self.dipole_errors 		= np.array(self.dipole_errors)
		self.tddft_angle_errors = np.array(self.tddft_angle_errors)
		self.xtb_angle_errors 	= np.array(self.xtb_angle_errors)

		self.tddft_energies *= 27.2114 #hartree to eV
		self.xtb_energies 	*= 27.2114 #hartree to eV
		self.energy_errors 	*= 27.2114 #hartree to eV

	def clean_by_Z_value(self, full_errors, mean, stddev):
		Z_value = lambda x : (x - mean)/stddev

		self.z_values = [Z_value(x) for x in full_errors.energy_errors]

		for enum, z in enumerate(self.z_values):
			if abs(z) < 2:
				self.chromophores.append(full_errors.chromophores[enum])
				self.Na_Ncs.append(full_errors.Na_Ncs[enum])
				self.tddft_energies.append(full_errors.tddft_energies[enum])
				self.xtb_energies.append(full_errors.xtb_energies[enum])
				self.tddft_dipoles.append(full_errors.tddft_dipoles[enum])
				self.xtb_dipoles.append(full_errors.xtb_dipoles[enum])
				self.tddft_dipole_mags.append(full_errors.tddft_dipole_mags[enum])
				self.xtb_dipole_mags.append(full_errors.xtb_dipole_mags[enum])
				self.tddft_angle_errors.append(full_errors.tddft_angle_errors[enum])
				self.xtb_angle_errors.append(full_errors.xtb_angle_errors[enum])

				self.energy_errors.append(full_errors.energy_errors[enum])
				self.dipole_errors.append(full_errors.dipole_errors[enum])

		self.tddft_energies 	= np.array(self.tddft_energies)
		self.xtb_energies 		= np.array(self.xtb_energies)
		self.energy_errors 		= np.array(self.energy_errors)
		self.dipole_errors 		= np.array(self.dipole_errors)
		self.tddft_angle_errors = np.array(self.tddft_angle_errors)
		self.xtb_angle_errors	= np.array(self.xtb_angle_errors)

	def __init__(self, results=None, full_errors=None, mean=None, stddev=None, with_outliers=True):
		self.chromophores 		= []
		self.Na_Ncs 			= []
		self.tddft_energies 	= []
		self.xtb_energies 		= []
		self.tddft_dipoles 		= []
		self.xtb_dipoles 		= []
		self.tddft_dipole_mags 	= []
		self.xtb_dipole_mags	= []
		self.energy_errors 		= []
		self.z_values 			= []
		self.dipole_errors 		= []
		self.tddft_angle_errors = []
		self.xtb_angle_errors	= []

		if with_outliers:
			self.make_full_error_lists(results)

		else:
			assert(full_errors is not None and mean is not None and stddev is not None)
			self.clean_by_Z_value(full_errors, mean, stddev)

		assert(len(self.tddft_energies) == len(self.chromophores))
		assert(len(self.tddft_energies) == len(self.Na_Ncs))
		assert(len(self.tddft_energies) == len(self.xtb_energies))
		assert(len(self.tddft_energies) == len(self.tddft_dipoles))
		assert(len(self.tddft_energies) == len(self.xtb_dipoles))
		assert(len(self.tddft_energies) == len(self.energy_errors))
		assert(len(self.tddft_energies) == len(self.dipole_errors))
		assert(len(self.tddft_energies) == len(self.tddft_angle_errors))
		assert(len(self.tddft_energies) == len(self.xtb_angle_errors))


class Results():
	def sanitize_results(self, results):
		results_dict = {}
		chromophores = []

		for i in results:
			c = i[0]
			xtb = i[1]

			chromophores.append(c)

			package = {
			"Na_Nc" : ref_data[c]["Na_Nc"],
			"tddft_energy" : ref_data[c]["energy"],
			"xtb_energy" : xtb[c]["excitation_energy"],
			"tddft_dipole" : ref_data[c]["transition_dipole"],
			"xtb_dipole" : xtb[c]["transition_dipole"]
			}

			for key, value in package.items(): 
				if package[key] is None:
					print(f"None value for {key} for chromophore {c}") 

					exit()

			results_dict[c] = package

		return chromophores, results_dict

	def make_dataframe(self):
		assert(self.training_set)

		set_type = ["test" for x in self.chromophores]

		for enum, i in enumerate(set_type):
			if self.chromophores[enum] in self.training_set:
				set_type[enum] = "training"


		import pandas as pd
		to_be_df = {	
			"chromophores" 		: self.chromophores,
			"set"				: set_type,
			"tddft_energy" 		: self.full.tddft_energies,
			"xtb_energy" 		: self.full.xtb_energies,
			"energy_error" 		: self.full.energy_errors,
			"Z_values"			: self.clean.z_values,
			"tddft_dipoles"		: self.full.tddft_dipoles,
			"xtb_dipoles"		: self.full.xtb_dipoles,
			"dipole_errors"		: self.full.dipole_errors,
			"Na_Nc" 			: self.full.Na_Ncs,
			"tddft_angle_errors": self.full.tddft_angle_errors,
			"xtb_angle_errors" 	: self.full.xtb_angle_errors,
		}

		df = pd.DataFrame.from_dict(to_be_df, orient='columns')

		return df


	def __init__(self, _results, training_set=[]):
		self.training_set = training_set
		self.chromophores, self.results = self.sanitize_results(_results)

		self.full = Errors(results=self.results, with_outliers=True)

		self.energy_mean = np.mean(self.full.energy_errors)
		self.energy_stddev = np.std(self.full.energy_errors)

		self.clean = Errors(full_errors=self.full, mean=self.energy_mean, stddev=self.energy_stddev, with_outliers=False)

		self.slope, self.intercept, self.r_value, self.p_value, self.stderr = linregress(self.clean.tddft_energies, self.clean.xtb_energies)

		self.energy_correlation = 1 - self.r_value**2
		self.energy_MAE = np.mean(abs(self.clean.energy_errors))
		self.dipole_MAE = np.mean(abs(self.clean.dipole_errors))



class Optimizer():

	#validation set data - do not use!
	validation_set = ['step_1601_chromophore_21', 'step_1451_chromophore_25', 'step_401_chromophore_19', 'step_701_chromophore_2',
 'step_1001_chromophore_7', 'step_1601_chromophore_15', 'step_1251_chromophore_21', 'step_1701_chromophore_16',
 'step_1001_chromophore_5', 'step_1551_chromophore_10', 'step_351_chromophore_1', 'step_1151_chromophore_15',
 'step_701_chromophore_11', 'step_1651_chromophore_26', 'step_851_chromophore_20', 'step_151_chromophore_9',
 'step_1451_chromophore_6', 'step_1101_chromophore_3', 'step_351_chromophore_23', 'step_1501_chromophore_10',
 'step_401_chromophore_3', 'step_1_chromophore_12', 'step_1751_chromophore_23', 'step_351_chromophore_3',
 'step_1151_chromophore_13', 'step_1351_chromophore_18', 'step_1601_chromophore_17', 'step_1701_chromophore_14',
 'step_1_chromophore_13', 'step_101_chromophore_18', 'step_951_chromophore_5', 'step_251_chromophore_19',
 'step_1251_chromophore_6', 'step_1651_chromophore_2', 'step_551_chromophore_12', 'step_701_chromophore_26',
 'step_651_chromophore_27', 'step_551_chromophore_10', 'step_1701_chromophore_1', 'step_401_chromophore_14',
 'step_851_chromophore_3', 'step_501_chromophore_22', 'step_451_chromophore_22', 'step_251_chromophore_15',
 'step_501_chromophore_8', 'step_1301_chromophore_5', 'step_851_chromophore_14', 'step_651_chromophore_11',
 'step_1401_chromophore_8', 'step_551_chromophore_24', 'step_1701_chromophore_12', 'step_101_chromophore_19',
 'step_251_chromophore_22', 'step_151_chromophore_8', 'step_1001_chromophore_27', 'step_551_chromophore_14',
 'step_1_chromophore_11', 'step_401_chromophore_2', 'step_901_chromophore_21', 'step_301_chromophore_13',
 'step_1101_chromophore_16', 'step_1601_chromophore_11', 'step_901_chromophore_9', 'step_1051_chromophore_13',
 'step_1801_chromophore_17', 'step_1_chromophore_17', 'step_1051_chromophore_10', 'step_1151_chromophore_8',
 'step_151_chromophore_3', 'step_451_chromophore_5', 'step_1001_chromophore_22', 'step_1851_chromophore_26',
 'step_1451_chromophore_23', 'step_1151_chromophore_17', 'step_401_chromophore_15', 'step_1451_chromophore_21',
 'step_1_chromophore_23', 'step_1551_chromophore_15', 'step_801_chromophore_13', 'step_1701_chromophore_6',
 'step_951_chromophore_7', 'step_901_chromophore_14', 'step_1801_chromophore_22', 'step_1351_chromophore_16',
 'step_1801_chromophore_21', 'step_401_chromophore_12', 'step_951_chromophore_11', 'step_1451_chromophore_26',
 'step_801_chromophore_2', 'step_1701_chromophore_10', 'step_1201_chromophore_12', 'step_1301_chromophore_13',
 'step_651_chromophore_21', 'step_101_chromophore_23', 'step_1151_chromophore_24', 'step_801_chromophore_1',
 'step_51_chromophore_3', 'step_1101_chromophore_11', 'step_1751_chromophore_18', 'step_751_chromophore_20',
 'step_1201_chromophore_27', 'step_101_chromophore_7', 'step_701_chromophore_23', 'step_1551_chromophore_1',
 'step_1251_chromophore_9', 'step_1751_chromophore_8', 'step_501_chromophore_18', 'step_51_chromophore_17',
 'step_901_chromophore_19', 'step_1151_chromophore_23', 'step_1801_chromophore_11', 'step_251_chromophore_27',
 'step_1451_chromophore_3', 'step_1001_chromophore_11', 'step_1701_chromophore_19', 'step_901_chromophore_22',
 'step_1001_chromophore_8', 'step_1501_chromophore_26', 'step_1751_chromophore_24', 'step_801_chromophore_19',
 'step_451_chromophore_16', 'step_901_chromophore_13', 'step_651_chromophore_18', 'step_1851_chromophore_13',
 'step_251_chromophore_12', 'step_851_chromophore_27', 'step_951_chromophore_25', 'step_1651_chromophore_4',
 'step_801_chromophore_26', 'step_1551_chromophore_11', 'step_1101_chromophore_20', 'step_51_chromophore_18',
 'step_751_chromophore_8', 'step_201_chromophore_23', 'step_1301_chromophore_2', 'step_1201_chromophore_11',
 'step_451_chromophore_10', 'step_201_chromophore_7', 'step_501_chromophore_27', 'step_1501_chromophore_17',
 'step_51_chromophore_14', 'step_1051_chromophore_26', 'step_1801_chromophore_16', 'step_701_chromophore_7',
 'step_451_chromophore_26', 'step_1751_chromophore_2', 'step_1401_chromophore_11', 'step_251_chromophore_2',
 'step_751_chromophore_22', 'step_1351_chromophore_22', 'step_1201_chromophore_8', 'step_651_chromophore_23',
 'step_1101_chromophore_15', 'step_901_chromophore_8', 'step_351_chromophore_4', 'step_501_chromophore_11',
 'step_1451_chromophore_17', 'step_1001_chromophore_2', 'step_1601_chromophore_1', 'step_1151_chromophore_21',
 'step_201_chromophore_22', 'step_851_chromophore_22', 'step_1851_chromophore_19', 'step_901_chromophore_15',
 'step_1651_chromophore_12', 'step_201_chromophore_5', 'step_801_chromophore_20', 'step_1351_chromophore_26',
 'step_101_chromophore_24', 'step_1301_chromophore_24', 'step_1851_chromophore_12', 'step_1351_chromophore_6',
 'step_151_chromophore_23', 'step_501_chromophore_19', 'step_1051_chromophore_16', 'step_1_chromophore_24',
 'step_601_chromophore_6', 'step_1151_chromophore_5', 'step_301_chromophore_16', 'step_301_chromophore_22',
 'step_401_chromophore_10', 'step_851_chromophore_21', 'step_1351_chromophore_3', 'step_1201_chromophore_10',
 'step_1351_chromophore_12', 'step_1_chromophore_16', 'step_401_chromophore_11', 'step_1651_chromophore_6',
 'step_1101_chromophore_17', 'step_1551_chromophore_19', 'step_1551_chromophore_3', 'step_451_chromophore_9',
 'step_101_chromophore_10', 'step_851_chromophore_11', 'step_1751_chromophore_25', 'step_751_chromophore_4',
 'step_201_chromophore_10', 'step_601_chromophore_7', 'step_851_chromophore_10', 'step_1351_chromophore_1',
 'step_151_chromophore_26', 'step_1151_chromophore_12', 'step_1801_chromophore_12', 'step_951_chromophore_9',
 'step_451_chromophore_21', 'step_801_chromophore_21', 'step_1351_chromophore_4', 'step_851_chromophore_13',
 'step_1101_chromophore_19', 'step_351_chromophore_2', 'step_1201_chromophore_5', 'step_1151_chromophore_11',
 'step_1251_chromophore_2', 'step_1201_chromophore_25', 'step_851_chromophore_6', 'step_201_chromophore_24',
 'step_1051_chromophore_11', 'step_1401_chromophore_14', 'step_451_chromophore_2', 'step_301_chromophore_5',
 'step_1051_chromophore_23', 'step_1301_chromophore_25', 'step_201_chromophore_9', 'step_1651_chromophore_19',
 'step_1751_chromophore_16', 'step_1301_chromophore_21', 'step_1401_chromophore_20', 'step_401_chromophore_9',
 'step_651_chromophore_14', 'step_1851_chromophore_2', 'step_601_chromophore_16', 'step_251_chromophore_7',
 'step_51_chromophore_27', 'step_201_chromophore_14', 'step_1501_chromophore_15', 'step_251_chromophore_10',
 'step_1751_chromophore_20', 'step_1251_chromophore_1', 'step_1251_chromophore_7', 'step_1101_chromophore_24',
 'step_1801_chromophore_7', 'step_1401_chromophore_7', 'step_601_chromophore_5', 'step_1701_chromophore_17',
 'step_1251_chromophore_11', 'step_1551_chromophore_25', 'step_1251_chromophore_12', 'step_351_chromophore_6',
 'step_301_chromophore_4', 'step_551_chromophore_20', 'step_51_chromophore_1', 'step_1551_chromophore_24',
 'step_851_chromophore_25', 'step_551_chromophore_26', 'step_801_chromophore_3', 'step_1451_chromophore_19',
 'step_451_chromophore_19', 'step_1601_chromophore_4', 'step_151_chromophore_7', 'step_1_chromophore_21',
 'step_1651_chromophore_23', 'step_1251_chromophore_24', 'step_1101_chromophore_14', 'step_1151_chromophore_27',
 'step_1251_chromophore_14', 'step_601_chromophore_26', 'step_1851_chromophore_6', 'step_1501_chromophore_16',
 'step_301_chromophore_18', 'step_1151_chromophore_6', 'step_451_chromophore_18', 'step_401_chromophore_17',
 'step_1151_chromophore_16', 'step_351_chromophore_19', 'step_1501_chromophore_7', 'step_1351_chromophore_24',
 'step_1001_chromophore_10', 'step_251_chromophore_16', 'step_1651_chromophore_5', 'step_451_chromophore_1',
 'step_1651_chromophore_22', 'step_1701_chromophore_22', 'step_1301_chromophore_10', 'step_1751_chromophore_13',
 'step_601_chromophore_2', 'step_451_chromophore_15', 'step_801_chromophore_6', 'step_801_chromophore_14',
 'step_1101_chromophore_8', 'step_1_chromophore_18', 'step_601_chromophore_10', 'step_601_chromophore_21',
 'step_851_chromophore_24', 'step_1201_chromophore_1', 'step_1301_chromophore_3', 'step_351_chromophore_18',
 'step_1701_chromophore_8', 'step_851_chromophore_15', 'step_1351_chromophore_10', 'step_1351_chromophore_23',
 'step_601_chromophore_12', 'step_1301_chromophore_1', 'step_1651_chromophore_18', 'step_1401_chromophore_10',
 'step_151_chromophore_22', 'step_1401_chromophore_18', 'step_51_chromophore_7', 'step_1701_chromophore_3',
 'step_1801_chromophore_9', 'step_1501_chromophore_2', 'step_701_chromophore_1', 'step_1551_chromophore_8',
 'step_1201_chromophore_2', 'step_951_chromophore_8', 'step_101_chromophore_17', 'step_601_chromophore_14',
 'step_1251_chromophore_18', 'step_1601_chromophore_12', 'step_701_chromophore_13', 'step_1251_chromophore_10',
 'step_1101_chromophore_13', 'step_1301_chromophore_22', 'step_551_chromophore_17', 'step_951_chromophore_15',
 'step_1051_chromophore_12', 'step_201_chromophore_4', 'step_1101_chromophore_2', 'step_1151_chromophore_7',
 'step_101_chromophore_22', 'step_401_chromophore_20', 'step_1301_chromophore_11', 'step_701_chromophore_6',
 'step_1651_chromophore_7', 'step_51_chromophore_25', 'step_1401_chromophore_15', 'step_1051_chromophore_6',
 'step_751_chromophore_17', 'step_1701_chromophore_11', 'step_651_chromophore_5', 'step_51_chromophore_22',
 'step_1401_chromophore_21', 'step_101_chromophore_5', 'step_1_chromophore_26', 'step_101_chromophore_25',
 'step_1601_chromophore_20', 'step_301_chromophore_7', 'step_1801_chromophore_4', 'step_1051_chromophore_9',
 'step_251_chromophore_20', 'step_251_chromophore_14', 'step_501_chromophore_4', 'step_751_chromophore_10',
 'step_1001_chromophore_19', 'step_1_chromophore_27', 'step_1801_chromophore_14', 'step_51_chromophore_8',
 'step_401_chromophore_27', 'step_951_chromophore_26', 'step_1851_chromophore_9', 'step_1201_chromophore_3',
 'step_1001_chromophore_20', 'step_1351_chromophore_7', 'step_1501_chromophore_5', 'step_1301_chromophore_16',
 'step_1301_chromophore_20', 'step_651_chromophore_1', 'step_1251_chromophore_13', 'step_1051_chromophore_5',
 'step_1351_chromophore_8', 'step_951_chromophore_13', 'step_1351_chromophore_20', 'step_901_chromophore_16',
 'step_1701_chromophore_4', 'step_551_chromophore_21', 'step_751_chromophore_12', 'step_501_chromophore_2',
 'step_801_chromophore_9', 'step_1051_chromophore_15', 'step_1001_chromophore_21', 'step_51_chromophore_5',
 'step_601_chromophore_24', 'step_301_chromophore_10', 'step_651_chromophore_10', 'step_401_chromophore_25',
 'step_1451_chromophore_10', 'step_1701_chromophore_21', 'step_1701_chromophore_24', 'step_301_chromophore_19',
 'step_1251_chromophore_15', 'step_601_chromophore_11', 'step_151_chromophore_17', 'step_301_chromophore_20',
 'step_851_chromophore_4', 'step_1051_chromophore_7', 'step_351_chromophore_22', 'step_151_chromophore_14',
 'step_1401_chromophore_12', 'step_1751_chromophore_9', 'step_751_chromophore_16', 'step_1101_chromophore_18',
 'step_601_chromophore_27', 'step_951_chromophore_6', 'step_1_chromophore_9', 'step_1251_chromophore_19',
 'step_1551_chromophore_6', 'step_251_chromophore_5', 'step_901_chromophore_17', 'step_601_chromophore_1',
 'step_151_chromophore_13', 'step_251_chromophore_23', 'step_1451_chromophore_16', 'step_901_chromophore_11',
 'step_451_chromophore_23', 'step_651_chromophore_22', 'step_1701_chromophore_5', 'step_1851_chromophore_15',
 'step_501_chromophore_20', 'step_401_chromophore_24', 'step_401_chromophore_18', 'step_1801_chromophore_2',
 'step_601_chromophore_19', 'step_1101_chromophore_1', 'step_1051_chromophore_21', 'step_851_chromophore_26',
 'step_401_chromophore_1', 'step_551_chromophore_18', 'step_501_chromophore_13', 'step_451_chromophore_6',
 'step_151_chromophore_6', 'step_1601_chromophore_22', 'step_1151_chromophore_19', 'step_1451_chromophore_5',
 'step_401_chromophore_23', 'step_651_chromophore_20', 'step_1501_chromophore_12', 'step_201_chromophore_13',
 'step_101_chromophore_9', 'step_1201_chromophore_14', 'step_901_chromophore_1', 'step_1851_chromophore_22',
 'step_951_chromophore_2', 'step_1701_chromophore_9', 'step_351_chromophore_14', 'step_201_chromophore_26',
 'step_901_chromophore_18', 'step_601_chromophore_20', 'step_51_chromophore_9', 'step_151_chromophore_5',
 'step_1051_chromophore_1', 'step_101_chromophore_27', 'step_101_chromophore_15', 'step_1201_chromophore_13',
 'step_801_chromophore_23', 'step_1401_chromophore_2', 'step_451_chromophore_11', 'step_201_chromophore_11',
 'step_1201_chromophore_19', 'step_751_chromophore_2', 'step_1651_chromophore_3', 'step_851_chromophore_2',
 'step_551_chromophore_25', 'step_1551_chromophore_5', 'step_1751_chromophore_3', 'step_451_chromophore_12',
 'step_1451_chromophore_18', 'step_901_chromophore_2', 'step_1501_chromophore_20', 'step_951_chromophore_27',
 'step_1251_chromophore_27', 'step_801_chromophore_11', 'step_1751_chromophore_26', 'step_1001_chromophore_14',
 'step_1851_chromophore_16', 'step_651_chromophore_13', 'step_1751_chromophore_7', 'step_701_chromophore_16',
 'step_1151_chromophore_26', 'step_51_chromophore_20', 'step_301_chromophore_26', 'step_1501_chromophore_21',
 'step_651_chromophore_6', 'step_351_chromophore_16', 'step_1551_chromophore_16', 'step_851_chromophore_23',
 'step_751_chromophore_7', 'step_801_chromophore_25', 'step_551_chromophore_7', 'step_751_chromophore_3',
 'step_951_chromophore_22', 'step_51_chromophore_6', 'step_1101_chromophore_12', 'step_1851_chromophore_24',
 'step_1801_chromophore_23', 'step_1601_chromophore_9', 'step_1301_chromophore_26', 'step_1151_chromophore_18',
 'step_451_chromophore_14', 'step_1501_chromophore_3', 'step_351_chromophore_12', 'step_851_chromophore_12',
 'step_1051_chromophore_3', 'step_1451_chromophore_7', 'step_801_chromophore_24', 'step_101_chromophore_26',
 'step_701_chromophore_17', 'step_1651_chromophore_17', 'step_651_chromophore_16']

	def make_active_param_list(self, active_params=[]):
		"""
		make the list of active parameters, being optimized

		>>> o.make_active_param_list(active_params=["k_s", "k_p", "k_d"])
		['k_s', 'k_p', 'k_d']

		>>> o.make_active_param_list([])
		['k_s', 'k_p', 'k_d', 'k_EN_s', 'k_EN_p', 'k_EN_d', 'k_T', 'Mg_s', 'Mg_p', 'Mg_d', 'N_s', 'N_p']

		"""
		all_params = ["k_s", "k_p", "k_d", "k_EN_s", "k_EN_p", "k_EN_d", "k_T", "Mg_s", "Mg_p", "Mg_d", "N_s", "N_p", "a_x", "y_J", "y_K"]
		if not active_params:
			return all_params
		else:
			assert(set(active_params).issubset(set(all_params)))
			assert(len(set(active_params)) == len(active_params))
				
			return active_params

	def make_initial_guess(self):
		GFN0_defaults = {
			"k_s" 		: 2.0,
			"k_p" 		: 2.48,
			"k_d" 		: 2.27,
			"k_EN_s" 	: 0.006,
			"k_EN_p" 	: -0.001,
			"k_EN_d"	: -0.002,
			"k_T" 		: 0.000,
			"Mg_s" 		: 1.0,
			"Mg_p" 		: 1.0,
			"Mg_d" 		: 1.0,
			"N_s" 		: 1.0, 
			"N_p" 		: 1.0, 
			"a_x"		: 0.5,
			"y_J"		: 4.0,
			"y_K"		: 2.0
		}
		
		return [GFN0_defaults[p] for p in self.active_params]


	"""
	store functions and data for optimization
	"""
	def __init__(self, samples, ref_data, method, active_params=[], max_iter=1):
		self.method = method
		self.ref_data = ref_data

		self.test_set = [x for x in list(self.ref_data.keys()) if x not in self.validation_set]
		assert(len(list(set(self.test_set).intersection(self.validation_set))) == 0)

		self.training_set = random.sample(self.test_set, k=samples)
		assert(len(self.training_set) == samples)
		assert(len(list(set(self.training_set).intersection(self.validation_set))) == 0)
		
		self.iter = 1
		
		self.active_params = self.make_active_param_list(active_params)
		self.initial_guess = self.make_initial_guess()

		self.max_iter = max_iter
		self.save = True
		self.start_time = datetime.datetime.now()
		self.time = time.time()

	def generate_result(self, input_tuple):
		chromophore, input_str = input_tuple

		result = run_qcore(input_str)

		return [chromophore, result]


	def generate_results(self, params, test=False):
		"""
		runs xtb for each chlorophyll molecule
		"""
		params_dict = dict(zip(self.active_params, params))

		input_str = "\"{chromophore} := bchla(structure(file = \'tddft_data/{chromophore}.xyz\') input_params={params})\""

		chromophores = self.training_set

		if test:
			chromophores = self.test_set

		input_strs = list(map(lambda x : input_str.format(chromophore=x, params=params_dict), chromophores))

		with ProcessPoolExecutor(max_workers=20) as pool:
			xtb_results = list(pool.map(self.generate_result, list(zip(chromophores, input_strs))))
		
		return Results(xtb_results, training_set=self.training_set)

	def step(self, params):
		results = self.generate_results(params)

		return results.energy_MAE + results.energy_correlation + results.dipole_MAE


	def param_string(self, params):
		"""
		writes the parameter string for logging and printing

		>>> o.param_string([1, 2, 3])
		'k_s : 1.000 k_p : 2.000 k_d : 3.000 '
		"""
		result = ""
		for enum, key in enumerate(self.active_params):
			result += "%s : %3.3f " % (key, params[enum])

		return result


	def callback(self, params, test=False):
		"""
		callback function to be run after iterations steps. Prints/stores the parameter strings
		along with the single fitness value.
		"""
		iter_str = "iter : {0:4d}".format(self.iter)
		
		params_list = None

		if type(params) == list:
			params_list = params
		else:
			params_list = params.tolist()

		param_str = self.param_string(params_list)

		results = self.generate_results(params, test)

		if test:
			assert(len(results.full.xtb_energies) == len(self.test_set))
		else:
			assert(len(results.full.xtb_energies) == len(self.training_set))

		fitness_str = "MAE(energy) : {0:3.3f} R^2 : {1:3.3f} ".format(results.energy_MAE, 1-results.energy_correlation)
		fitness_str += f"MAE(dipole) : {results.dipole_MAE:3.3f}"

		time_str = "time/s : {0:3.6f}".format(time.time() - self.time)
		self.time = time.time()

		log_string = "{iter_} {param} {fitness} {time}".format(iter_=iter_str,
									param=param_str,
									fitness=fitness_str,
									time=time_str)

		print(log_string)
		
		self.iter += 1

		return 

	def make_step_function(self):
		"""
		lambda wrapper for scipy optimize
		"""
		return lambda x : self.step(x)
		

	def optimize(self):
		"""
		run the optimization method
		"""
		if self.method =="test":
			from scipy.optimize import rosen, rosen_der
			x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
			return minimize(
				rosen,
				x0,
				method="Nelder-Mead",
				tol=1e-6,
				options={"disp" : True}
			)

		if self.method == "Nelder-Mead":
			return minimize(
			fun=self.make_step_function(), 
			x0=self.initial_guess, 
			callback=self.callback,
			method="Nelder-Mead",
			options={"maxiter" : self.max_iter+1, "disp": True, "adaptive" : True}
			)

		elif self.method == "Bayesian_Gaussian_Process":
			from skopt import gp_minimize
			return gp_minimize(
			self.make_fitness_function,
			dimensions=[
			(0, 3.0), #k_s
			(0, 3.0), #k_p
			(0, 3.0), #k_d
			(-0.01, 0.01), #k_EN_s
			(-0.01, 0.01), #k_EN_p
			(-0.01, 0.01), #k_EN_d
			(0.0, 0.5), #k_T
			(0.0, 5.0), #Mg_S
			(0.0, 5.0), #Mg_p
			(0.0, 5.0), #Mg_d
			(0.0, 5.0), #N_s
			(0.0, 5.0), #N_p
			],
			n_calls=self.max_iter,
			callback=self.callback,
			n_initial_points=100,
			x0=[
			2.0, #k_S
			2.48, #k_P
			2.27, #k_D
			0.006, #k_EN_s
			-0.001, #k_EN_p
			-0.002, #k_EN_d
			0.000, #k_T
			1.0, #Mg_s
			1.0, #Mg_p
			1.0, #Mg_d
			1.0,  #N_s
			1.0,  #N_p
			]
			)

	def test_result(self, params):
		"""
		given a list of parameters, will run the full test set. 
		DIFFERENT to the validation set!
		"""
		if self.method == "test":
			print(params)
			return

		train_results = self.generate_results(params)
		test_results = self.generate_results(params, test=True)

		no_test_set_samples = len(test_results.full.xtb_energies)

		print(f"# of test set samples: {no_test_set_samples}")

		print("test set results:")
		fitness_str = "MAE(energy) : {0:3.3f} R^2 : {1:3.3f} ".format(test_results.energy_MAE, 1-test_results.energy_correlation)
		fitness_str += f"MAE(dipole) : {test_results.dipole_MAE:3.3f}"

		fig1, ax1 = plt.subplots()
		fig2, ax2 = plt.subplots()
		fig3, ax3 = plt.subplots(subplot_kw={'projection': 'polar'})

		ax1.scatter(test_results.clean.tddft_energies, test_results.clean.xtb_energies, label="test set", color='black', marker='x')
		ax1.scatter(train_results.clean.tddft_energies, train_results.clean.xtb_energies, label="training set", color='red', marker='x')
		ax1.set_xlabel("TD-DFT excitation energies / eV")
		ax1.set_ylabel("xtb excitation energies / eV")

		ax2.scatter(test_results.clean.tddft_dipole_mags, test_results.clean.xtb_dipole_mags, label="test set", color='black', marker='x')
		ax2.scatter(train_results.clean.tddft_dipole_mags, train_results.clean.xtb_dipole_mags, label="training set", color='red', marker='x')
		ax2.set_xlabel("TD-DFT $|\mu|$")
		ax2.set_ylabel("xtb $|\mu|$")

		ax3.scatter(np.deg2rad(test_results.clean.tddft_angle_errors), test_results.clean.tddft_dipole_mags, label="TDDFT", color='black')
		ax3.scatter(np.deg2rad(train_results.clean.tddft_angle_errors), train_results.clean.tddft_dipole_mags, label="TDDFT", color='red')
		ax3.scatter(np.deg2rad(test_results.clean.xtb_angle_errors), test_results.clean.xtb_dipole_mags,  label="xtb", color='black', marker='x')
		ax3.scatter(np.deg2rad(train_results.clean.xtb_angle_errors), train_results.clean.xtb_dipole_mags,  label="xtb", color='red', marker='x')
		ax3.set_thetamin(0)
		ax3.set_thetamax(30)
		ax3.set_ylim([0,6])

		fig1.set_size_inches(12, 12)
		fig2.set_size_inches(12, 12)
		fig3.set_size_inches(12, 12)

		ax1.legend()
		ax2.legend()
		ax3.legend()
		
		import pickle as pkl

		pkl.dump(fig1, open("EnergiesScatter.pkl", 'wb'))
		pkl.dump(fig2, open("DipoleMagsScatter.pkl", 'wb'))
		pkl.dump(fig3, open("AnglesScatter.pkl", 'wb'))

		df = test_results.make_dataframe()

		with open("Results.tex", 'w') as tex_file:
			print(df.to_latex(index=False), file=tex_file)

		pkl.dump(df, open("Test_results.pkl", 'wb'))


if __name__ == '__main__':
	args = CLI.parse_args()
	try:
		if args.run_tests:
			ref_data = make_ref_data(args.ref_data)
			print("running doctests")
			import doctest
			doctest.testmod(verbose=True, extraglobs={'o' : Optimizer(ref_data, method='testing', active_params=['k_s', 'k_p', 'k_d'], max_iter=50)})
			print("doctests finished")
			exit(0)
	
	except:
		pass
	else:
		print()
		print("#######################")
		print("# BChla-xTB optimizer #")
		print("#######################")

		print()

		start = time.time()
		print("start time: ", time.ctime())

		print()
		
		active_params = args.params
		print("active parameters from python argument input : ", end="")
		print(active_params)
		print()

		#construct reference data
		ref_data = make_ref_data(args.ref_data)
		print("reference data constructed from : \"%s\"" % args.ref_data)
		print()

		#make optimizer
		method   = args.method[0]
		max_iter = args.max_iter
		samples  = args.samples[0]

		print("Optimization method : ", method)
		print("maximum iterations : ", max_iter)
		print("# of training set samples: ", samples)

		print()
		print("recreate input with:")
		print("python optimizer.py", end=" ")
		print("--params %s" % " ".join(args.params), end=" ")
		print("--samples %s" % samples, end=" ")
		print("--method %s" % method, end=" ")
		print("--max_iter %i" % max_iter, end=" ")
		print("--ref_data %s" % args.ref_data , end=" ")
		print("--run_tests %r" % args.run_tests, end=" ")
		print()

		print("making optimizer...")
		optimizer = Optimizer(ref_data=ref_data, samples=samples, method=method, active_params=active_params, max_iter=max_iter)
		print()
		#run optimization
		print("running optimization...")
		print()
		optimizer_result = optimizer.optimize()
		
		print()
		optimized_params = [round(x, 3) for x in optimizer_result.x]

		if method == "test":
			zipped_params = dict(zip(["x1", "x2", "x3", "x4", "x5"], optimized_params))
			print("optimized parameters: ", zipped_params)
		else:
			zipped_params = dict(zip(args.params, optimized_params))
			print("optimized parameters: ", zipped_params)
		print()


		#run validation
		print("running validation...")
		optimizer.test_result(optimized_params)

		print()
		print(f"wall-clock time : {time.time() - start:6.3f} seconds")
		print()
		print("#######################")


