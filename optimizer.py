import scipy.optimize
from scipy.optimize import minimize
from scipy.stats import linregress

import enum
import argparse
import subprocess
import json
import numpy as np
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
	"--weights",
	nargs=3,
	type=float,
	default=[1., 1., 1.],
	help="""the weights associated with the MAE of the excitation energy, R^2 values for excitation energy and
transition dipole magnitude in the objective function""")

CLI.add_argument(
	"--method",
	nargs=1,
	type=str,
	default=["SLSQP"],
	choices=["SLSQP", "test", "Bayesian_Gaussian_Process"],
	help="specify optimization method, or flag to run a validation set"
)

CLI.add_argument(
	"--max_iter",
	nargs=1,
	type=int,
	default=[5000],
	help="maximum number of iterations for optimization method"
)

CLI.add_argument(
	"--ref_data",
	nargs=1,
	type=str,
	default=['tddft_data/tddft_data.json'],
	help="json file that stores reference data, used to optimize against"
)

CLI.add_argument(
	"--run_tests",
	nargs=1,
	type=bool,
	default=False,
	help="bool to flag running doctests instead of an optimization"
)

CLI.add_argument(
	"--name",
	nargs=1,
	type=str,
	help="name for the output files",
	required=True
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


def run_qcore(input_tuple):
	"""
	runs qcore with the input string
	"""
	
	chromophore, chromophore_str = input_tuple

	qcore_path = "~/.local/src/Qcore/release/qcore"
	#qcore_path = "~/qcore/cmake-build-release/bin/qcore"
	json_str = " -n 1 -f json --schema none -s "
	norm_str = " -n 1 -s "

	json_run = subprocess.run(qcore_path + json_str + chromophore_str,
					shell=True,
					stdout=subprocess.PIPE,
					executable="/bin/bash",
					universal_newlines=True)

	try:
		json_results = json.loads(json_run.stdout)

	except:
		print(qcore_path + norm_str + chromophore_str)

		norm_run = subprocess.run(qcore_path + norm_str + chromophore_str,
					shell=True,
					stdout=subprocess.PIPE,
					executable="/bin/bash",
					universal_newlines=True)

		print(norm_run.stdout)

	if json_results[chromophore]["excitation_energy"] is None or json_results[chromophore]["transition_dipole"] is None:
		print(qcore_path + norm_str + chromophore_str)

		norm_run = subprocess.run(qcore_path + norm_str + chromophore_str,
					shell=True,
					stdout=subprocess.PIPE,
					executable="/bin/bash",
					universal_newlines=True)

		print(norm_run.stdout)

	return [chromophore, json_results]

class Errors():
	def make_full_error_lists(self, results):
		for i in results.values():
			angle_error = calc_angle_error(i["tddft_dipole"], i["xtb_dipole"])
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

		_, _, self.energy_r_value, _, _ = linregress(self.clean.tddft_energies, self.clean.xtb_energies)
		_, _, self.dipole_r_value, _, _ = linregress(self.clean.tddft_dipole_mags, self.clean.xtb_dipole_mags)

		self.energy_correlation = 1 - self.energy_r_value**2
		self.dipole_correlation = 1 - self.dipole_r_value**2
		self.energy_RMSE = np.sqrt(np.mean(np.square(self.clean.energy_errors)))
		#self.dipole_mag_RMSE = np.sqrt(np.mean(np.square(self.clean.xtb_dipole_mags - self.clean.tddft_dipole_mags)))



class Optimizer():
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

	def make_bounds(self):
		bounds = {
			"k_s" 		: (None, None),
			"k_p" 		: (None, None),
			"k_d" 		: (None, None),
			"k_EN_s" 	: (None, None),
			"k_EN_p" 	: (None, None),
			"k_EN_d"	: (None, None),
			"k_T" 		: (None, None),
			"Mg_s" 		: (None, None),
			"Mg_p" 		: (None, None),
			"Mg_d" 		: (None, None),
			"N_s" 		: (None, None), 
			"N_p" 		: (None, None), 
			"a_x"		: (0, None),
			"y_J"		: (0, None),
			"y_K"		: (0, None)
		}

		return [bounds[p] for p in self.active_params]


	def read_sets(self):
		training_set = []
		test_set = []
		validation_set = []

		with open("/home/of15641/chlorophyll_parameterization/training_set.txt") as training_set_file:
		#with open("training_set.txt") as training_set_file:
			for line in training_set_file.readlines():
				training_set.append(line.replace("\n", ""))

		with open("/home/of15641/chlorophyll_parameterization/test_set.txt") as test_set_file:
		#with open("test_set.txt") as test_set_file:
			for line in test_set_file.readlines():
				test_set.append(line.replace("\n", ""))

		with open("/home/of15641/chlorophyll_parameterization/validation_set.txt") as validation_set_file:
		#with open("validation_set.txt") as validation_set_file:
			for line in validation_set_file.readlines():
				validation_set.append(line.replace("\n", ""))

		assert(set(training_set).issubset(set(list(self.ref_data.keys()))))
		assert(set(test_set).issubset(set(list(self.ref_data.keys()))))
		assert(set(validation_set).issubset(set(list(self.ref_data.keys()))))
		assert(set(training_set).issubset(set(test_set)))
		assert(len(list(set(test_set).intersection(validation_set))) == 0)
		assert(len(list(set(training_set).intersection(validation_set))) == 0)
		assert(len(training_set) == 100)

		return (training_set, test_set, validation_set)
		

	def __init__(self, ref_data, method, output_func, name, active_params=[], max_iter=1, weights=[1., 1., 1.]):
		self.name = name
		self.output = output_func

		self.method = method
		self.ref_data = ref_data
		self.weights = weights

		self.training_set, self.test_set, self.validation_set = self.read_sets()
		
		self.iter = 0
		
		self.active_params = self.make_active_param_list(active_params)
		self.initial_guess = self.make_initial_guess()
		self.bounds = self.make_bounds()

		self.max_iter = max_iter
		self.save = True
		self.start_time = datetime.datetime.now()
		self.time = time.time()

	def generate_results(self, params, test=False):
		"""
		runs xtb for each chlorophyll molecule
		"""
		params_dict = dict(zip(self.active_params, params))

		input_str = "\"{chromophore} := bchla(structure(file = \'/home/of15641/chlorophyll_parameterization/tddft_data/{chromophore}/{chromophore}.xyz\') input_params={params})\""
		#input_str = "\"{chromophore} := bchla(structure(file = \'tddft_data/{chromophore}/{chromophore}.xyz\') input_params={params})\""

		chromophores = self.training_set

		if test:
			chromophores = self.test_set

		input_strs = list(map(lambda x : input_str.format(chromophore=x, params=params_dict), chromophores))

		with ProcessPoolExecutor(max_workers=20) as pool:
			xtb_results = list(pool.map(run_qcore, list(zip(chromophores, input_strs))))
		
		return Results(xtb_results, training_set=self.training_set)

	def objective_function(self, params):
		results = self.generate_results(params)

		return self.weights[0] * results.energy_correlation + self.weights[1] * results.dipole_correlation + self.weights[2] * results.energy_RMSE


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

		fitness_str = "RMSE(energy) : {0:3.3f} R^2(energy) : {1:3.3f} ".format(results.energy_RMSE, 1-results.energy_correlation)
		fitness_str += f"R^2(dipole_mags) : {1-results.dipole_correlation:3.3f}"

		time_str = "time/s : {0:3.6f}".format(time.time() - self.time)
		self.time = time.time()

		log_string = "{iter_} {param} {fitness} {time}".format(iter_=iter_str,
									param=param_str,
									fitness=fitness_str,
									time=time_str)

		self.output(log_string)
		
		self.iter += 1

		return 
	

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

		if self.method == "SLSQP":
			self.callback(params=self.initial_guess)

			return minimize(
			fun=self.objective_function, 
			x0=self.initial_guess, 
			callback=self.callback,
			method="SLSQP",
			bounds=self.bounds,
			options={"maxiter" : self.max_iter+1}
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
		test_results  = self.generate_results(params, test=True)

		no_test_set_samples = len(test_results.full.xtb_energies)

		self.output(f"# of test set samples: {no_test_set_samples}")

		self.output("training set results:")

		training_fitness_str = "RMSE(energy) : {0:3.3f} R^2(energy) : {1:3.3f} ".format(train_results.energy_RMSE, 1-train_results.energy_correlation)
		training_fitness_str += f"R^2(dipole_mags) : {1-train_results.dipole_correlation:3.3f}"
		self.output(training_fitness_str)

		self.output("")


		self.output("test set results:")

		test_fitness_str = "RMSE(energy) : {0:3.3f} R^2(energy) : {1:3.3f} ".format(test_results.energy_RMSE, 1-test_results.energy_correlation)
		test_fitness_str += f"R^2(dipole_mags) : {1-test_results.dipole_correlation:3.3f}"
		self.output(test_fitness_str)

		self.output("")

		
		import pickle as pkl
		df = test_results.make_dataframe()

		#with open("results.tex", 'w') as tex_file:
		#	print(df.to_latex(index=False), file=tex_file)

		if self.name.endswith(".out"):
			pkl.dump(df, open(self.name.replace("out", "pkl"), 'wb'))
		else:
			pkl.dump(df, open(self.name + ".pkl", 'wb'))

def make_output_func(file_name):
	if file_name.endswith(".out"):
		file = open(file_name, 'w')
		return lambda x : print(x, file=file)
	else:
		file =  open(file_name+".out", 'w')
		return lambda x : print(x, file=file)

if __name__ == '__main__':
	args = CLI.parse_args()

	name = args.name[0]

	output = make_output_func(name)

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

		output("""
#######################
# BChla-xTB optimizer #
#######################
""")

		start = time.time()
		output(f"start time: {time.ctime()}")

		output("")
		
		active_params = args.params
		output(f"active parameters from python argument input : {active_params}")
		output("")

		#construct reference data
		ref_data = make_ref_data(args.ref_data[0])
		output("reference data constructed from : \"%s\"" % args.ref_data[0])
		output("")

		#make optimizer
		method   = args.method[0]
		max_iter = args.max_iter[0]
		weights =  args.weights

		output(f"Optimization method : {method}")
		output(f"maximum iterations : {max_iter}")

		output(f"""weights:
RMSE(energy): {weights[0]}
R^2 (energy): {weights[1]}
R^2 (dipole): {weights[2]}
""")

		output(f"""recreate input with:
python optimizer.py \
--params {" ".join(args.params)} \
--method {method} \
--max_iter {max_iter} \
--ref_data {args.ref_data[0]} \
--run_tests {args.run_tests} \
--weights {" ".join([str(w) for w in weights])} \
""")
		output("making optimizer...")
		optimizer = Optimizer(ref_data=ref_data, 
							method=method,
			 				active_params=active_params,
			  				max_iter=max_iter,
			   				weights=weights,
			    			output_func=output,
			    			name=name
		)

		output("")

	
		#run optimization
		output("running optimization...")
		output("")
		optimizer_result = optimizer.optimize()

		output(f"""
{optimizer_result.message}
Current function value: {optimizer_result.fun:3.3f}
Iterations: {optimizer_result.nit}
Function evaluations: {optimizer_result.nfev}
Gradient evaluations: {optimizer_result.njev}
""")
		optimized_params = [round(x, 3) for x in optimizer_result.x]

		if method == "test":
			zipped_params = dict(zip(["x1", "x2", "x3", "x4", "x5"], optimized_params))
			output(f"optimized parameters: {zipped_params}")
		else:
			zipped_params = dict(zip(args.params, optimized_params))
			output(f"optimized parameters: {zipped_params}")
		output("")


		#run validation
		output("running validation...")
		optimizer.test_result(optimized_params)

		output(f"""
wall-clock time : {time.time() - start:6.3f} seconds

#######################
""")


