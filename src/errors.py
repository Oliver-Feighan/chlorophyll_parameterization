from utils import calc_angle_error, calc_dipole_error
import numpy as np

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


