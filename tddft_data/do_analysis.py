import matplotlib.pyplot as plt
import numpy as np
import re
import json
import os

##Rip transition properties from output files. If transition properties don't exist, 
##	add the xyz file to the re-run pile.
molecules = os.listdir(path='.')

#coordinate lines for the N_A, N_C atoms 
Na_line = 18
Nc_line = 35

output_files = []
xyz_files = []

for i in molecules:
	if i.endswith(".out"):
		output_files.append(i)

	elif i.endswith(".xyz"):
		xyz_files.append(i)

results = {}

rerun_pile = []

no_homo_lumo_trans = []

for file in output_files:
	# last number should be the transition dipole line
	excited_states = []

	lines = list(open(file, 'r'))
	for num, line in enumerate(lines):
		if "Excited State" in line or "Transition Dipole Moments" in line:
			excited_states.append(num)

	if not excited_states:
		rerun_pile.append(file)
		continue

	transition_dipole_start = excited_states[-1]

	homo_lumo_coeff = 0
	homo_lumo_energy = 0
	homo_lumo_trans = 0

	for state in range(len(excited_states)-1):
		state_lines = lines[excited_states[state]:excited_states[state+1]]
		for line in state_lines:
			if "->" in line:
				mo_coeff = re.findall(r'-?\d+\.?\d+', line)
				occ, virt, coeff = mo_coeff
				if int(virt) - int(occ) == 1:
					if abs(float(coeff)) > homo_lumo_coeff:
						homo_lumo_coeff = float(coeff)
						homo_lumo_trans = state

						energy_hr, energy_eV, S_2, Multiplicity = re.findall(r'-?\d+\.\d+', state_lines[0])

						homo_lumo_energy = float(energy_hr)

	transition_dipole_line = transition_dipole_start + 2 + homo_lumo_trans
	x, y, z = re.findall(r'-?\d*\.\d*', lines[transition_dipole_line])

	assert(homo_lumo_energy != 0)

	xyz_file = file.replace('.out', '.xyz')
	
	coords = list(open(xyz_file))

	Na = np.array([float(x) for x in re.findall(r'-?\d+\.\d+', coords[Na_line])])
	Nc = np.array([float(x) for x in re.findall(r'-?\d+\.\d+', coords[Nc_line])])

	Na_Nc = Na-Nc

	results[file.replace(".out", "")] = {
			"energy" : homo_lumo_energy, 
			"transition_dipole" : [float(x), float(y), float(z)],
			"Na_Nc" : Na_Nc.tolist()
	}

## Analysis
energies = []
Na_Nc_lengths = []
transition_dipole_mags = []
angle_errors = []

def angle_error(vec1, vec2):
	numerator = np.dot(vec1, vec2)
	denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

	return np.rad2deg(np.arccos(numerator/denominator))

def angle_error_with_phase(vec1, vec2):
	return min(angle_error(vec1, vec2), angle_error(-vec1, vec2))

for chrom in results.values():
	Na_Nc_lengths.append(np.linalg.norm(np.array(chrom["Na_Nc"])))
	transition_dipole_mags.append(np.linalg.norm(chrom["transition_dipole"]))
	energies.append(chrom["energy"])
	angle_errors.append(angle_error_with_phase(np.array(chrom["transition_dipole"]), np.array(chrom["Na_Nc"])))

def plot_histogram(values, title):
	fig, ax = plt.subplots()
	ax.set_title(title)
	ax.hist(values, bins=30)

plot_histogram(np.array(energies), "energies")
plot_histogram(np.array(Na_Nc_lengths), "$N_A$-$N_C$ lengths")
plot_histogram(np.array(transition_dipole_mags), "transition dipone magnitudes")
plot_histogram(np.array(angle_errors), "angle error to $Q_y$ axis")

plt.show()

rerun_pile_file = open('rerun_pile.txt', 'w')
for i in rerun_pile:
	print(i, file=rerun_pile_file)

with open('tddft_data.json', 'w') as dump_file:
	json.dump(results, sort_keys=True, indent=4, fp=dump_file)











					





