import scipy.optimize
from scipy.optimize import minimize

import subprocess
import random
import json
import numpy as np
import random
import time
import datetime
import sys
from concurrent.futures import ProcessPoolExecutor

def angle_error(vector_1, vector_2):
	"""
	computes the angle between two transition dipoles.

	>>> angle_error([0,0,1], [0,1,1])
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


def generate_result(input_tuple):
	chromophore, input_str = input_tuple

	try:
		result = run_qcore(input_str)
	except:
		return None
	else:
		return [chromophore, result]

def generate_results(ref_data, params):
	"""
	runs exc-xtb for each chlorophyll molecule, sanitizing results

	>>> ref_data = make_ref_data()
	>>> keys = list(ref_data.keys())[:3]
	>>> ref_data = {keys[0] : ref_data[keys[0]], keys[1] : ref_data[keys[1]], keys[2] : ref_data[keys[2]]}
	>>> test=generate_results(ref_data, params=[1.827, 3.089, 1.531, -0.029, 0.013, 0.001, 0.0])
	>>> test
	{'step_1_chromophore_01': {'tddft_energy': 0.067717, 'xtb_energy': 0.029292343886254457, 'energy_error': 0.03842465611374554, 'tddft_dipole': [0.305422706, -2.699393089, 0.410750669], 'xtb_dipole': [0.5124307476124165, -4.498082315965987, 0.218572342605373], 'dipole_error': 5.8340260552349354}, 'step_1_chromophore_02': {'tddft_energy': 0.069685, 'xtb_energy': 0.031652590077129616, 'energy_error': 0.03803240992287038, 'tddft_dipole': [-2.43839633, 0.294525704, -0.899224337], 'xtb_dipole': [-4.022174469548795, 0.8850977027546133, -1.60640001070345], 'dipole_error': 5.303504105038609}, 'step_1_chromophore_03': {'tddft_energy': 0.069472, 'xtb_energy': 0.03186119181447111, 'energy_error': 0.03761080818552889, 'tddft_dipole': [0.108963122, -2.698943706, -0.009963017], 'xtb_dipole': [-0.2097896114822028, 4.360348315008915, -0.5581703233391183], 'dipole_error': 7.51076667604526}}
	"""
	params_dict = dict(zip(["k_s", "k_p", "k_d", "k_EN_s", "k_EN_p", "k_EN_d", "k_T"], params))

	#qcore_path = "/Users/of15641/qcore/cmake-build-debug/bin/qcore"
	qcore_path = "~/.local/src/Qcore/release/qcore"
	input_str = ' -n 1 -f json -s "{chromophore} := excited_scf(structure(file = \'xyz_files/{chromophore}.xyz\') xtb(model=\'gfn0\' input_params={params}))" '

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
			"energy_error" : ref_data[c]["energy"] - xtb[c]["excitation_energy"],
			"tddft_dipole" : ref_data[c]["transition_dipole"],
			"xtb_dipole" : xtb[c]["transition_dipole"],
			"dipole_error" : angle_error(ref_data[c]["transition_dipole"], xtb[c]["transition_dipole"])
			}

			results[c] = package

	return results


class Optimizer():
	"""
	store functions and data for optimization
	>>> ref_data=make_ref_data()
	>>> keys = list(ref_data.keys())[:3]
	>>> ref_data = {keys[0] : ref_data[keys[0]], keys[1] : ref_data[keys[1]], keys[2] : ref_data[keys[2]]}
	>>> test=Optimizer(ref_data)
	>>> test.run_opt()
	iter :    2 k_S : 1.827 k_P : 3.089 k_D : 1.531 k_EN_s : -0.029 k_EN_p : 0.013 k_EN_d : 0.001 k_T : 0.000 fitness : 1.034
	"""
	test_set = [
						'step_101_chromophore_10', 'step_151_chromophore_02', 'step_251_chromophore_09', 'step_151_chromophore_16', 'step_351_chromophore_09',
						'step_301_chromophore_22', 'step_51_chromophore_19', 'step_351_chromophore_15', 'step_351_chromophore_21', 'step_501_chromophore_18',
						'step_1_chromophore_21', 'step_301_chromophore_02', 'step_301_chromophore_05', 'step_451_chromophore_22', 'step_251_chromophore_06',
						'step_251_chromophore_02', 'step_101_chromophore_04', 'step_1_chromophore_10', 'step_351_chromophore_13', 'step_51_chromophore_18',
						'step_1_chromophore_26', 'step_251_chromophore_21', 'step_351_chromophore_05', 'step_101_chromophore_26', 'step_101_chromophore_02',
						'step_151_chromophore_04', 'step_451_chromophore_27', 'step_301_chromophore_03', 'step_201_chromophore_01', 'step_351_chromophore_04',
						'step_451_chromophore_14', 'step_101_chromophore_08', 'step_301_chromophore_24', 'step_301_chromophore_10', 'step_201_chromophore_26',
						'step_101_chromophore_05', 'step_301_chromophore_04', 'step_401_chromophore_05', 'step_51_chromophore_10', 'step_101_chromophore_22',
						'step_451_chromophore_26', 'step_451_chromophore_09', 'step_101_chromophore_15', 'step_201_chromophore_14', 'step_151_chromophore_09',
						'step_451_chromophore_15', 'step_151_chromophore_25', 'step_501_chromophore_23', 'step_201_chromophore_25', 'step_101_chromophore_17',
						'step_351_chromophore_01', 'step_351_chromophore_17', 'step_301_chromophore_20', 'step_401_chromophore_09', 'step_501_chromophore_14',
						'step_401_chromophore_11', 'step_451_chromophore_01', 'step_51_chromophore_26', 'step_1_chromophore_22', 'step_51_chromophore_21',
						'step_201_chromophore_05', 'step_1_chromophore_01', 'step_451_chromophore_03', 'step_201_chromophore_27', 'step_101_chromophore_12',
						'step_351_chromophore_03', 'step_1_chromophore_03', 'step_301_chromophore_19', 'step_501_chromophore_20', 'step_501_chromophore_06',
						'step_101_chromophore_25', 'step_451_chromophore_07', 'step_101_chromophore_11', 'step_301_chromophore_17', 'step_301_chromophore_27',
						'step_201_chromophore_21', 'step_501_chromophore_17', 'step_401_chromophore_17', 'step_101_chromophore_20', 'step_501_chromophore_10',
						'step_301_chromophore_11', 'step_151_chromophore_22', 'step_151_chromophore_05', 'step_151_chromophore_14', 'step_151_chromophore_24',
						'step_51_chromophore_04', 'step_201_chromophore_10', 'step_1_chromophore_05', 'step_51_chromophore_16', 'step_451_chromophore_06',
						'step_51_chromophore_22', 'step_401_chromophore_20', 'step_351_chromophore_10', 'step_251_chromophore_17', 'step_351_chromophore_25',
						'step_251_chromophore_19', 'step_301_chromophore_15', 'step_101_chromophore_14', 'step_1_chromophore_25', 'step_101_chromophore_09'
	]

	def make_test_set(self, ref_data):
		result = {}

		if self.method == "validate":
			return ref_data

		for c in list(ref_data.keys()):
			if c in self.test_set:
				result[c] = ref_data[c]

		return result


	def __init__(self, method, ref_data=None, max_iter=1):
		self.method = method
		self.ref_data = self.make_test_set(ref_data)
		self.iter = 1
		self.initial_guess = {
			"k_S" : 1.827,
			"k_P" : 3.089,
			"k_D" : 1.531,
			"k_EN_s" : -0.029,
			"k_EN_p" : 0.013,
			"k_EN_d" : 0.001,
			"k_T" : 0.0,
		}

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

	def fitness_function(self, results):
		energy_errors = []
		angle_errors  = []

		for i in results.values():
			 energy_errors.append(i["energy_error"])
			 angle_errors.append(i["dipole_error"])

		energy_errors = np.array(energy_errors)
		energy_errors *= 27.2114 #hartree to eV
		mean = np.mean(np.abs(energy_errors))

		return mean		

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

		return self.fitness_function(results)


	def callback(self, params):
		iter_str = "iter : {0:4d}".format(self.iter)
		
		param_str = "k_s : {0:3.3f} \
k_p : {1:3.3f} \
k_d : {2:3.3f} \
k_EN_s : {3:3.3f} \
k_EN_p : {4:3.3f} \
k_EN_d : {5:3.3f} \
k_T : {6:3.3f}".format(*params.tolist())
		
		results = generate_results(self.ref_data, params)
		fitness = self.fitness_function(results)

		fitness_str = "fitness : {0:3.3f}".format(fitness)

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

			with open(self.file_name_str + ".json", "a") as json_dump:
				json.dump(results, json_dump)
		
		self.iter += 1

		return 

	def make_fitness_function(self):
		"""
		lambda wrapper for scipy optimize
		"""
		return lambda x : self.step(x)
		

	def run(self):
		IG_as_list = list(self.initial_guess.values())

		if self.method == "optimize":
			minimize(
			self.make_fitness_function(), 
			IG_as_list, 
			callback=self.callback,
			method="Nelder-Mead",
			options={"maxiter" : self.max_iter+1, "adaptive" : True}
			)

		elif self.method == "validate":
			self.callback(np.array(IG_as_list))



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
	optimizer = Optimizer("optimize", ref_data, 500)
	#optimizer = Optimizer("validate", ref_data, 500)
	optimizer.run()

