# List of all active parameters in the optimization method.
all_params = ["k_s", "k_p", "k_d",
              "k_EN_s", "k_EN_p", "k_EN_d",
              "k_T",
              "Mg_s", "Mg_p", "Mg_d", "N_s", "N_p", "Mgs_Ns", "Mgs_Np", "Mgp_Ns", "Mgp_Np",
              "a_x", "y_J", "y_K",
              "E_Mg_s", "E_Mg_p", "E_Mg_d",
              "level_shift",
              "D_scl"]

# Default values for parameters in the GFN1-xTB method
GFN1_defaults = {
    "k_s" : 1.602,
    "k_p" 		: 3.328,
    "k_d" 		: 2.00,
    "k_EN_s" 	: 0.006,
    "k_EN_p" 	: -0.001,
    "k_EN_d"	: -0.002,
    "k_T" 		: 0.000,
    "Mg_s" 		: 1.152,
    "Mg_p" 		: 1.217,
    "N_s" 		: 1.088,
    "N_p" 		: 0.895,

    "Mgs_Ns"	: 1.0,
    "Mgs_Np"	: 1.0,
    "Mgp_Ns"	: 1.0,
    "Mgp_Np"	: 1.0,

    "a_x"		: 0.053,
    "y_J"		: 1.945,
    "y_K"		: 3.95,
    "E_Mg_s"	: 0.0,
    "E_Mg_p" 	: 0.0,
    "E_Mg_d"	: 0.0,
    "level_shift"	: 0.0,
    "D_scl"		: 0.5
}

# Default values for parameters in the GFN0-xTB method
GFN0_defaults = {
    "k_s" 	: 2.0,
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
    "y_K"		: 2.0,
    "E_Mg_s"	: 0.0,
    "E_Mg_p" 	: 0.0,
    "E_Mg_d"	: 0.0
}

# Bounds for parameters when SLSQP method is used
bounds = {
    "k_s" 	: (None, None),
    "k_p" 		: (None, None),
    "k_d" 		: (None, None),
    "k_EN_s" 	: (None, None),
    "k_EN_p" 	: (None, None),
    "k_EN_"	: (None, None),
    "k_T" 		: (None, None),
    "Mg_s" 		: (None, None),
    "Mg_p" 		: (None, None),
    "Mg_d" 		: (None, None),
    "N_s" 		: (None, None),
    "N_p" 		: (None, None),
    "Mgs_Ns"    : (None, None),
    "Mgs_Np"    : (None, None),
    "Mgp_Ns"    : (None, None),
    "Mgp_Np"    : (None, None),
    "a_x"		: (0, None),
    "y_J"		: (0, None),
    "y_K"		: (0, None),
    "E_Mg_s" 	: (None, None),
    "E_Mg_p" 	: (None, None),
    "E_Mg_d" 	: (None, None),
    "level_shift" 	: (0, None),
    "D_scl" : (0, None)
}

# Bounds for parameters when Bayesian Gaussian method is used
bayesian_dimensions = [
			(0, 3.0),  # k_s
			(0, 3.0),  # k_p
			(0, 3.0),  # k_d
			(-0.01, 0.01),  # k_EN_s
			(-0.01, 0.01),  # k_EN_p
			(-0.01, 0.01),  # k_EN_d
			(0.0, 0.5),  # k_T
			(0.0, 5.0),  # Mg_S
			(0.0, 5.0),  # Mg_p
			(0.0, 5.0),  # Mg_d
			(0.0, 5.0),  # N_s
			(0.0, 5.0),  # N_p
			]

# Initial Guess for Bayesian Gaussian method
x0 = [
2.0,  # k_S
2.48,  # k_P
2.27,  # k_D
0.006,  # k_EN_s
-0.001,  # k_EN_p
-0.002,  # k_EN_d
0.000,  # k_T
1.0,  # Mg_s
1.0,  # Mg_p
1.0,  # Mg_d
1.0,  # N_s
1.0,  # N_p
]
