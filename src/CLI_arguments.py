import argparse

def make_CLI():
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--params",
        nargs="*",
        type=str,
        default=[],
        help="active parameters for optimization",
    )

    CLI.add_argument(
        "--weights",
        nargs=4,
        type=float,
        default=[1., 1., 1., 1.],
        help="""the weights associated with the RMSE of the excitation energy, RMSE of the transition dipole magnitude, R^2 values for excitation energy and
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
        "--gfn",
        nargs=1,
        type=str,
        default=['gfn1'],
        help="the gfn theory used. Both GFN1 and GFN0 will use eigdiff and MNOK correction."
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

    return CLI
