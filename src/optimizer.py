import scipy.optimize
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from scipy.stats import linregress

# local files
import utils
from results import Results
from parameters import *

# external modules
import time
import datetime

class Optimizer():
	def make_active_param_list(self, active_params=[]):
		"""
		make the list of active parameters, being optimized

		>>> o.make_active_param_list(active_params=["k_s", "k_p", "k_d"])
		['k_s', 'k_p', 'k_d']

		>>> o.make_active_param_list([])
		['k_s', 'k_p', 'k_d', 'k_EN_s', 'k_EN_p', 'k_EN_d', 'k_T', 'Mg_s', 'Mg_p', 'Mg_d', 'N_s', 'N_p']

		"""

		if not active_params:
			return all_params
		else:
			assert(set(active_params).issubset(set(all_params)))
			assert(len(set(active_params)) == len(active_params))
				
			return active_params

	def make_initial_guess(self):
		if self.gfn == "gfn0":	
			return [GFN0_defaults[p] for p in self.active_params]
		elif self.gfn == "gfn1":
			return [GFN1_defaults[p] for p in self.active_params]

	def make_bounds(self):


		return [bounds[p] for p in self.active_params]


	def read_sets(self, training_set_filename, test_set_filename, validation_set_filename):
		training_set = []
		test_set = []
		validation_set = []

		with open(training_set_filename) as training_set_file:
			for line in training_set_file.readlines():
				training_set.append(line.replace("\n", ""))

		with open(test_set_filename) as test_set_file:
			for line in test_set_file.readlines():
				test_set.append(line.replace("\n", ""))

		with open(validation_set_filename) as validation_set_file:
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
		

	def __init__(self, **kwargs):
		self.name = kwargs.get("name")
		self.output = kwargs.get("output_func")

		self.method = kwargs.get("method") if "method" in kwargs else "SLSQP"
		self.gfn 	= kwargs.get("gfn") if "gfn" in kwargs else "GFN1"
		self.ref_data = kwargs.get("ref_data")
		self.weights = kwargs.get("weights") if "weights" in kwargs else [1., 1., 1., 1.]

		self.training_set, self.test_set, self.validation_set = self.read_sets(
			training_set_filename = kwargs.get("training_set_filename"),
			test_set_filename = kwargs.get("test_set_filename"),
			validation_set_filename = kwargs.get("validation_set_filename")
		)
		
		self.iter = 0
		
		self.active_params = self.make_active_param_list(kwargs.get("active_params")) if "active_params" in kwargs else []
		self.initial_guess = self.make_initial_guess()
		self.bounds = self.make_bounds()

		self.max_iter = kwargs.get("max_iter") if "max_iter" in kwargs else 5000
		self.save = True
		self.start_time = datetime.datetime.now()
		self.time = time.time()

	def generate_results(self, params, test=False):
		"""
		runs xtb for each chlorophyll molecule
		"""
		params_dict = dict(zip(self.active_params, params))
		
		level_shift = 0.
		
		if "level_shift" in self.active_params:
			level_shift = params_dict["level_shift"]
			del params_dict["level_shift"] 

		input_str = "\"{chromophore} := xtb(structure(file = \'/home/of15641/chlorophyll_parameterization/tddft_data/{chromophore}/{chromophore}.xyz\') level_shift={level_shift} model='chlorophyll' input_params={params})\""

		chromophores = self.training_set

		if test:
			chromophores = self.test_set

		input_strs = list(map(lambda x : input_str.format(chromophore=x, gfn=self.gfn, level_shift=level_shift, params=params_dict), chromophores))

		with ProcessPoolExecutor(max_workers=20) as pool:
			xtb_results = list(pool.map(utils.run_qcore, list(zip(chromophores, input_strs))))
		
		return Results(xtb_results, ref_data=self.ref_data, training_set=self.training_set)

	def objective_function(self, params):
		results = self.generate_results(params)

		return self.weights[0] * results.energy_RMSE + self.weights[1] * (1 - results.energy_correlation) + self.weights[2] * results.dipole_mag_RMSE + self.weights[3] * (1 - results.dipole_correlation)


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

		#if test:
			#assert(len(results.full.xtb_energies) == len(self.test_set))
		#else:
			#assert(len(results.full.xtb_energies) == len(self.training_set))

		fitness_str = "RMSE(energy) : {0:3.3f} R^2(energy) : {1:3.3f} ".format(results.energy_RMSE, results.energy_correlation)
		fitness_str += f"RMSE(dipole_mags) : {results.dipole_mag_RMSE:3.3f} R^2(dipole_mags) : {results.dipole_correlation:3.3f} "
		fitness_str += f"fitness : {self.weights[0] * (1 - results.energy_RMSE) + self.weights[1] * (1 - results.energy_correlation) + self.weights[2] * results.dipole_mag_RMSE + self.weights[3] * (1 - results.dipole_correlation)}"

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
			dimensions=bayesian_dimensions,
			n_calls=self.max_iter,
			callback=self.callback,
			n_initial_points=100,
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

		training_fitness_str = "RMSE(energy) : {0:3.3f} R^2(energy) : {1:3.3f} ".format(train_results.energy_RMSE, train_results.energy_correlation)
		training_fitness_str += f"RMSE(dipole mags) : {train_results.dipole_mag_RMSE:3.3f} R^2(dipole_mags) : {train_results.dipole_correlation:3.3f} "
		training_fitness_str += f"fitness : {self.weights[0] * train_results.energy_RMSE + self.weights[1] * (1 - train_results.energy_correlation) + self.weights[2] * train_results.dipole_mag_RMSE + self.weights[3] * (1 - train_results.dipole_correlation)}"

		self.output(training_fitness_str)

		self.output("")


		self.output("test set results:")

		test_fitness_str = "RMSE(energy) : {0:3.3f} R^2(energy) : {1:3.3f} ".format(test_results.energy_RMSE, test_results.energy_correlation)
		test_fitness_str += f"RMSE(dipole_mags) : {test_results.dipole_mag_RMSE:3.3f} R^2(dipole_mags) : {test_results.dipole_correlation:3.3f} "
		test_fitness_str += f"fitness : {self.weights[0] * test_results.energy_RMSE + self.weights[1] * (1 - test_results.energy_correlation) + self.weights[2] * test_results.dipole_mag_RMSE + self.weights[3] * (1 - test_results.dipole_correlation)}"

		self.output(test_fitness_str)

		self.output("")

		import pickle as pkl
		df = test_results.make_dataframe()

		if self.name.endswith(".out"):
			pkl.dump(df, open(self.name.replace("out", "pkl"), 'wb'))
		else:
			pkl.dump(df, open(self.name + ".pkl", 'wb'))



