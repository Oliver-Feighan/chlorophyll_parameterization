from itertools import chain, combinations

all_params = ["k_s", "k_p", "k_d", "k_EN_s", "k_EN_p", "k_EN_d", "k_T", "Mg_s", "Mg_p", "Mg_d", "N_s", "N_p", "a_x", "y_J", "y_K"]

param_types = {
	"huckel" : ["k_s", "k_p", "k_d"],
	"electronegativity" : ["k_EN_s", "k_EN_p", "k_EN_d"],
	"kinetic" : ["k_T"],
	"global_pair" : ["Mg_s", "Mg_p", "Mg_d", "N_s", "N_p"],
	"response" : ["a_x", "y_J", "y_K"],
}

def powerset(params):
	"""
	generate the powerset from a given list of params
	>>> powerset(["k_s", "k_p", "k_d"])
	[('k_s',), ('k_p',), ('k_d',), ('k_s', 'k_p'), ('k_s', 'k_d'), ('k_p', 'k_d'), ('k_s', 'k_p', 'k_d')]
	"""

	return list(chain.from_iterable(combinations(params, r) for r in range(1, len(params)+1)))

def make_params_list(types_list):
	"""
	take list of parameter types and make the concatonated list of all specified parameters

	>>> make_params_list(('electronegativity', 'kinetic', 'response'))
	['k_EN_s', 'k_EN_p', 'k_EN_d', 'k_T', 'a_x', 'y_J', 'y_K']
 
 	>>> make_params_list(('huckel', 'electronegativity', 'kinetic', 'global_pair', 'response'))
	['k_s', 'k_p', 'k_d', 'k_EN_s', 'k_EN_p', 'k_EN_d', 'k_T', 'Mg_s', 'Mg_p', 'Mg_d', 'N_s', 'N_p', 'a_x', 'y_J', 'y_K']
	
	"""
	global param_types

	return list(chain.from_iterable([param_types[x] for x in types_list]))



if __name__ == "__main__":
	all_types_powerset = powerset(list(param_types.keys()))

	script_template = """#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=24:mem=10gb
#PBS -N {name}
#PBS -j oe

cd ~/chlorophyll_parameterization

module load lang/python/anaconda/3.8-2020.07

export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=TBB

python optimizer.py --params {params}

	"""

	for combination in all_types_powerset:
		params_list = make_params_list(combination)
		name = "_".join(combination)

		with open(f"{name}.sub", 'w') as f:
			print(script_template.format(name=name, params=" ".join(params_list)), file=f)








