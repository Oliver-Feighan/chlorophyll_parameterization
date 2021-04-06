import scipy.optimize
from scipy.optimize import minimize
from scipy.stats import linregress

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

CLI=argparse.ArgumentParser()
CLI.add_argument(
	"--params",
	nargs="*",
	type=str,
	default=[],
	help="active parameters for optimization",
)

CLI.add_argument(
	"--method",
	nargs=1,
	type=str,
	default="Nelder-Mead",
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
	default='tddft_data/tddft_results.json',
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


def run_qcore(input_str):
	"""
	runs qcore with the input string
	"""
	return json.loads(subprocess.run(input_str,
					shell=True,
					stdout=subprocess.PIPE,
					executable="/bin/bash",
					universal_newlines=True).stdout)

class Optimizer():
	def make_active_param_list(self, active_params=[]):
		"""
		make the list of active parameters, being optimized

		>>> o.make_active_param_list(active_params=["k_s", "k_p", "k_d"])
		['k_s', 'k_p', 'k_d']

		>>> o.make_active_param_list([])
		['k_s', 'k_p', 'k_d', 'k_EN_s', 'k_EN_p', 'k_EN_d', 'k_T', 'Mg_s', 'Mg_p', 'Mg_d', 'N_s', 'N_p']

		"""
		all_params = ["k_s", "k_p", "k_d", "k_EN_s", "k_EN_p", "k_EN_d", "k_T", "Mg_s", "Mg_p", "Mg_d", "N_s", "N_p"]
		if not active_params:
			return all_params
		else:
			assert(set(active_params).issubset(set(all_params)))
			assert(len(set(active_params)) == len(active_params))
				
			return active_params

	"""
	store functions and data for optimization
	"""
	def __init__(self, ref_data, method, active_params=[], max_iter=1):
		self.method = method
		self.ref_data = ref_data
		
		self.test_set = random.sample(list(self.ref_data.keys()), k=100)
		self.iter = 1
		
		#defaults GFN0
		self.initial_guess = {
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
		}
		
		self.active_params = self.make_active_param_list(active_params)

		self.max_iter = max_iter
		self.log = []
		self.save = True
		self.start_time = datetime.datetime.now()
		self.time = time.time()
		self.file_name_str = "{method}_{year}_{month}_{day}_{hour}{minute}".format(
			method = method,
            year = self.start_time.year,
            month = self.start_time.month,
            day = self.start_time.day,
            hour = self.start_time.hour,
            minute = self.start_time.minute
        )

	def generate_result(self, input_tuple):
		chromophore, input_str = input_tuple

		result = run_qcore(input_str)
		return [chromophore, result]


	def generate_results(self, params):
		"""
		runs exc-xtb for each chlorophyll molecule, sanitizing results
		"""
		params_dict = dict(zip(self.active_params, params))

		qcore_path = "/Users/of15641/qcore/cmake-build-release/bin/qcore"
		#qcore_path = "~/.local/src/Qcore/release/qcore"
		input_str = ' -n 1 -f json --schema none -s "{chromophore} := bchla(structure(file = \'xyz_files/{chromophore}.xyz\') input_params={params})" '

		chromophores = list(ref_data.keys())
		input_strs = list(map(lambda x : qcore_path + input_str.format(chromophore=x, params=params_dict), chromophores))

		with ProcessPoolExecutor(max_workers=20) as pool:
			xtb_results = list(pool.map(self.generate_result, list(zip(chromophores, input_strs))))
		
		results = {}

		for i in xtb_results:
				c = i[0]
				xtb = i[1]

				package = {
				"tddft_energy" : ref_data[c]["energy"],
				"xtb_energy" : xtb[c]["excitation_energy"],
				"tddft_dipole" : ref_data[c]["transition_dipole"],
				"xtb_dipole" : xtb[c]["transition_dipole"]
				}

				results[c] = package

		return results

	def fitness_function(self, results):
		"""
		Parses results from generate results. Appends all errors into lists for error
		calculation.

		>>> test_results = { \
"step_1_chromophore_1" : { \
"tddft_energy" : 1.0, \
"xtb_energy" : 1.5, \
"tddft_dipole" : [0., 1., 1.], \
"xtb_dipole" : [0., 3., 3.] \
}, \
"step_1_chromophore_2" : { \
"tddft_energy" : 1.0, \
"xtb_energy" : 1.0, \
"tddft_dipole" : [0., 1., 1.], \
"xtb_dipole" : [0., 1., 1.] \
} \
}
		>>> o.fitness_function(test_results)				
		(6.80285, 1.0, 1.4142135623730951)

		"""
		tddft_energies = []
		xtb_energies = [] 
		energy_errors = []
		dipole_errors = []
		angle_errors  = []

		for i in results.values():
			angle_error = calc_angle_error(i["tddft_dipole"], i["xtb_dipole"])
			if angle_error < 20:
				tddft_energies.append(i["tddft_energy"])
				xtb_energies.append(i["xtb_energy"])
				energy_errors.append(i["xtb_energy"] - i["tddft_energy"])
				dipole_errors.append(calc_dipole_error(i["xtb_dipole"], i["tddft_dipole"]))
				angle_errors.append(angle_error)

		slope, intercept, r_value, p_value, std_err = linregress(xtb_energies, tddft_energies)

		energy_errors = np.array(energy_errors)
		energy_errors *= 27.2114 #hartree to eV

		dipole_errors = np.array(dipole_errors)

		energy_MAE = np.mean(abs(energy_errors))

		energy_correlation = 1 - r_value**2

		dipole_MAE = np.mean(abs(dipole_errors))

		return (energy_MAE, energy_correlation, dipole_MAE)

	def step(self, params):
		"""	
		run the fitness function, and give back single fitness value
		"""
		results = self.generate_results(params)

		energy_MAE, energy_correlation, dipole_MAE = self.fitness_function(results)

		return energy_MAE + energy_correlation + dipole_MAE


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


	def callback(self, params):
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

		param_str = self.params_string(params_list)

		results = self.generate_results(params)
		MAE, correlation = self.fitness_function(results)

		fitness_str = "MAE : {0:3.3f} R^2 : {1:3.3f}".format(MAE, 1-correlation)

		time_str = "time/s : {0:3.6f}".format(time.time() - self.time)
		self.time = time.time()

		log_string = "{iter_} {param} {fitness} {time}\n".format(iter_=iter_str,
									param=param_str,
									fitness=fitness_str,
									time=time_str)

		self.log.append(log_string)

		print(log_string)
		
		self.iter += 1

		return 

	def make_fitness_function(self):
		"""
		lambda wrapper for scipy optimize
		"""
		return lambda x : self.step(x)
		

	def optimize(self):
		"""
		run the optimization method
		"""
		IG_as_list = list(self.initial_guess.values())

		if self.method == "Nelder-Mead":
			return minimize(
			self.make_fitness_function(), 
			IG_as_list, 
			callback=self.callback,
			method="Nelder-Mead",
			options={"maxiter" : self.max_iter+1, "adaptive" : True}
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
		self.callback(np.array(params))



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
		method   = args.method
		max_iter = args.max_iter

		print("Optimization method : ", method)
		print("maximum iterations : ", max_iter)

		print()
		print("recreate input with:")
		print("python optimizer.py", end=" ")
		print("--params %s" % " ".join(args.params), end=" ")
		print("--method %s" % method, end=" ")
		print("--max_iter %i" % max_iter, end=" ")
		print("--ref_data %s" % args.ref_data , end=" ")
		print("--run_tests %r" % args.run_tests, end=" ")
		print()

		print("making optimizer...")
		optimizer = Optimizer(ref_data=ref_data, method=method, active_params=active_params, max_iter=max_iter)
		print()
		#run optimization
		print("running optimization...")
		print()
		optimized_params = optimizer.optimize()

		#run validation
		#Optimizer.test_result(optimized_params)


