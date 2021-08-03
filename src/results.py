from errors import Errors

import numpy as np

from scipy.stats import linregress

class Results():
    def sanitize_results(self, results, ref_data):
        results_dict = {}
        chromophores = []

        for i in results:
            if not i[2]:
                continue

            c = i[0]
            xtb = i[1]

            chromophores.append(c)

            package = {
                "Na_Nc": ref_data[c]["Na_Nc"],
                "tddft_energy": ref_data[c]["energy"],
                "xtb_energy": xtb[c]["excitation_energy"],
                "tddft_dipole": ref_data[c]["transition_dipole"],
                "xtb_dipole": xtb[c]["transition_dipole"]
            }

            for key, value in package.items():
                if package[key] is None:
                    print(f"None value for {key} for chromophore {c}")

                    exit()

            results_dict[c] = package

        return chromophores, results_dict

    def make_dataframe(self):
        assert (self.training_set)

        set_type = ["test" for x in self.chromophores]

        for enum, i in enumerate(set_type):
            if self.chromophores[enum] in self.training_set:
                set_type[enum] = "training"

        import pandas as pd
        to_be_df = {
            "chromophores" 	: self.chromophores,
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


    def __init__(self, _results, ref_data, training_set=[]):
        self.training_set = training_set
        self.chromophores, self.results = self.sanitize_results(_results, ref_data)

        if len(self.chromophores) == 0 or len(list(self.results.keys())) == 0:
            self.energy_RMSE = 1
            self.energy_correlation = 0
            self.dipole_correlation = 0
            return

        self.full = Errors(results=self.results, with_outliers=True)

        self.energy_mean = np.mean(self.full.energy_errors)
        self.energy_stddev = np.std(self.full.energy_errors)

        self.clean = Errors(full_errors=self.full, mean=self.energy_mean, stddev=self.energy_stddev
                            , with_outliers=False)

        _, _, self.energy_r_value, _, _ = linregress(self.clean.tddft_energies, self.clean.xtb_energies)
        _, _, self.dipole_r_value, _, _ = linregress(self.clean.tddft_dipole_mags, self.clean.xtb_dipole_mags)

        self.energy_RMSE = np.sqrt(np.mean(np.square(self.clean.energy_errors)))
        self.dipole_mag_RMSE = np.sqrt(np.mean(np.square(self.clean.xtb_dipole_mags - self.clean.tddft_dipole_mags)))
        self.energy_correlation = self.energy_r_value**2
        self.dipole_correlation = self.dipole_r_value**2


