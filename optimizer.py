import scipy.optimize
from scipy.optimize import minimize
from scipy.stats import linregress

import subprocess
import random
import json
import numpy as np
import random
import time
import datetime
import sys
from concurrent.futures import ProcessPoolExecutor

def calc_angle_error(vector_1, vector_2):
	"""
	computes the angle between two transition dipoles.

	>>> calc_angle_error([0,0,1], [0,1,1])
	45.00000000000001
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


def make_ref_data():
	"""
	make dictionary object of the TD-DFT reference data

	>>> data = make_ref_data()
	>>> data["step_1_chromophore_01"]["energy"]
	0.067717
	>>> data["step_1_chromophore_01"]["transition_dipole"]
	[0.305422706, -2.699393089, 0.410750669]
	
	>>> type(data)
	<class 'dict'>

	"""
	data = open("tddft_results.json")
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
	def make_active_param_list(self, active_params=""):
		all_params = ["k_s", "k_p", "k_d", "k_EN_s", "k_EN_p", "k_EN_d", "k_T", "Mg_s", "Mg_p", "Mg_d", "N_s", "N_p"]
		if active_params == "":
			return all_params
		else:
			assert(set(active_params).issubset(all_params))
			assert(len(set(active_params)) == len(all_params))
			return active_params

	"""
	store functions and data for optimization
	>>> ref_data=make_ref_data()
	>>> keys = list(ref_data.keys())[:3]
	>>> ref_data = {keys[0] : ref_data[keys[0]], keys[1] : ref_data[keys[1]], keys[2] : ref_data[keys[2]]}
	>>> test=Optimizer(ref_data)
	>>> test.run_opt()
	iter :    2 k_S : 1.827 k_P : 3.089 k_D : 1.531 k_EN_s : -0.029 k_EN_p : 0.013 k_EN_d : 0.001 k_T : 0.000 fitness : 1.034
	"""
	def __init__(self, method, active_params="", ref_data=None, max_iter=1):
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

		try:
			result = run_qcore(input_str)
		except:
			return None
		else:
			return [chromophore, result]


	def generate_results(self, params):
		"""
		runs exc-xtb for each chlorophyll molecule, sanitizing results

		>>> ref_data = make_ref_data()
		>>> keys = list(ref_data.keys())[:3]
		>>> ref_data = {keys[0] : ref_data[keys[0]], keys[1] : ref_data[keys[1]], keys[2] : ref_data[keys[2]]}
		>>> test=generate_results(ref_data, params=[1.827, 3.089, 1.531, -0.029, 0.013, 0.001, 0.0])
		>>> test
		{'step_1_chromophore_01': {'tddft_energy': 0.067717, 'xtb_energy': 0.029292343886254457, 'energy_error': 0.03842465611374554, 'tddft_dipole': [0.305422706, -2.699393089, 0.410750669], 'xtb_dipole': [0.5124307476124165, -4.498082315965987, 0.218572342605373], 'dipole_error': 5.8340260552349354}, 'step_1_chromophore_02': {'tddft_energy': 0.069685, 'xtb_energy': 0.031652590077129616, 'energy_error': 0.03803240992287038, 'tddft_dipole': [-2.43839633, 0.294525704, -0.899224337], 'xtb_dipole': [-4.022174469548795, 0.8850977027546133, -1.60640001070345], 'dipole_error': 5.303504105038609}, 'step_1_chromophore_03': {'tddft_energy': 0.069472, 'xtb_energy': 0.03186119181447111, 'energy_error': 0.03761080818552889, 'tddft_dipole': [0.108963122, -2.698943706, -0.009963017], 'xtb_dipole': [-0.2097896114822028, 4.360348315008915, -0.5581703233391183], 'dipole_error': 7.51076667604526}}
		"""
		params_dict  = dict(zip([self.active_params], params))

		#qcore_path = "/Users/of15641/qcore/cmake-build-debug/bin/qcore"
		qcore_path = "~/.local/src/Qcore/release/qcore"
		input_str = ' -n 1 -f json --schema none -s "{chromophore} := bchla(structure(file = \'xyz_files/{chromophore}.xyz\') input_params={params})" '

		chromophores = list(ref_data.keys())
		input_strs = list(map(lambda x : qcore_path + input_str.format(chromophore=x, params=params_dict), chromophores))

		with ProcessPoolExecutor(max_workers=20) as pool:
			xtb_results = list(pool.map(generate_result, list(zip(chromophores, input_strs))))
		
		results = {}

		for i in xtb_results:
			if i is not None:
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

	def dipole_error(result):
		xtb = np.array(i['xtb_dipole'])
		tddft = np.array(i['tddft_dipole'])

		phase1 = np.linalg.norm(xtb - tddft)
		phase2 = np.linalg.norm(xtb - tddft)

		return min(phase1, phase2)

	def fitness_function(self, results):
		tddft_energies = []
		xtb_energies = [] 
		energy_errors = []
		dipole_errors = []
		angle_errors  = []

		for i in results.values():
			angle_error = calc_angle_error(i["tddft_dipole"], i["xtb_dipole"])
			if i["dipole_error"] < 20:
				tddft_energies.append(i["tddft_energy"])
				xtb_energies.append(i["xtb_energy"])
				energy_errors.append(i["energy_error"])
				dipole_errors.append(dipole_error(i))
				angle_errors.append(i["dipole_error"])

		slope, intercept, r_value, p_value, std_err = linregress(xtb_energies, tddft_energies)

		energy_errors = np.array(energy_errors)
		energy_errors *= 27.2114 #hartree to eV
		energy_MAE = np.mean(abs(energy_errors))

		energy_correlation = 1 - r_value**2

		dipole_MAE = np.mean(abs(dipole_errors))

		return (energy_MAE, energy_correlation, dipole_MAE)

	def step(self, params):
		"""	
	 	>>> ref_data=make_ref_data()
		>>> keys = list(ref_data.keys())[:3]
		>>> ref_data = {keys[0] : ref_data[keys[0]], keys[1] : ref_data[keys[1]], keys[2] : ref_data[keys[2]]}
		>>> test=Optimizer(ref_data)
		>>> test.step([1.827, 3.089, 1.531, -0.029, 0.013, 0.001, 0.0])
		1.0346488508694904
		"""
		results = generate_results(self.ref_data, params)

		energy_MAE, energy_correlation, dipole_MAE = self.fitness_function(results)

		return energy_MAE + energy_correlation + dipole_MAE


	def param_string(self, params):
		"""

		"""
		result = ""
		for enum, key in enumerate(self.active_params.keys()):
			result += "%s : %3.3f" % (key, params[enum])

		return result


	def callback(self, params):
		iter_str = "iter : {0:4d}".format(self.iter)
		
		params_list = None

		if type(params) == list:
			params_list = params
		else:
			params_list == *params.tolist()

		params_str = self.params_string(params_list)

		results = generate_results(self.ref_data, params)
		MAE, correlation = self.fitness_function(results)

		fitness_str = "MAE : {0:3.3f} R^2 : {1:3.3f}".format(MAE, 1-correlation)

		time_str = "time/s : {0:3.6f}".format(time.time() - self.time)
		self.time = time.time()

		log_string = "{iter_} {param} {fitness} {time}\n".format(iter_=iter_str,
									param=param_str,
									fitness=fitness_str,
									time=time_str)

		self.log.append(log_string)

		if self.save:
			with open(self.file_name_str + ".txt", "a") as output_file:
				output_file.write(log_string)
		
		self.iter += 1

		return 

	def make_fitness_function(self):
		"""
		lambda wrapper for scipy optimize
		"""
		return lambda x : self.step(x)
		

	def optimize(self):
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

	def validate_result(self, params):
			self.callback(np.array(params))



if __name__ == '__main__':
	try:
		if sys.argv[1] == "test":
			print("running doctests")
			import doctest
			doctest.testmod()
			print("doctests finished")
			exit(0)
	
	except:
		pass

	#construct reference data
	ref_data = make_ref_data()

	#make optimizer
	#optimizer = Optimizer("Nelder-Mead", ref_data, 5000)
	optimizer = Optimizer("Nelder-Mead", ref_data, 5000)

	#run optimization
	optimized_params = optimizer.run()

	#run validation
	Optimizer.validate_result(optimized_params)


