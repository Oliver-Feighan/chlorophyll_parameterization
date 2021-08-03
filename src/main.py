import CLI_arguments
import utils

from optimizer import Optimizer

import time

if __name__ == '__main__':
    reference_data_filename = utils.find("tddft_data.json", "../")
    training_set_filename = utils.find("training_set.txt", "../")
    test_set_filename = utils.find("test_set.txt", "../")
    validation_set_filename = utils.find("validation_set.txt", "../")

    print(reference_data_filename)
    print(training_set_filename)
    print(test_set_filename)
    print(validation_set_filename)

    CLI = CLI_arguments.make_CLI()
    args = CLI.parse_args()

    name = args.name[0]

    output = utils.make_output_func(name)

    try:
        if args.run_tests:
            ref_data = utils.make_ref_data(args.ref_data)
            print("running doctests")
            import doctest

            doctest.testmod(verbose=True, extraglobs={
                'o': Optimizer(ref_data, method='testing', active_params=['k_s', 'k_p', 'k_d'], max_iter=50)})
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

        # construct reference data
        ref_data_name = args.ref_data[0] if args.ref_data is not None else reference_data_filename
        ref_data = utils.make_ref_data(ref_data_name)
        output(f"reference data constructed from : \"{ref_data_name}\"")
        output(f"training set data file : {training_set_filename}")
        output(f"testing set data file : {test_set_filename}")
        output(f"validation set data file : {validation_set_filename}")
        output("")

        # make optimizer
        method = args.method[0]
        gfn = args.gfn[0]
        max_iter = args.max_iter[0]
        weights = args.weights

        output(f"Optimization method : {method}")
        output(f"GFN-xTB method : {gfn}")
        output(f"maximum iterations : {max_iter}")

        output(f"""weights:
RMSE(energy): {weights[0]}
RMSE(dipole): {weights[0]}
R^2 (energy): {weights[1]}
R^2 (dipole): {weights[2]}
""")

        output(f"""recreate input with:
python optimizer.py \
--params {" ".join(args.params)} \
--method {method} \
--max_iter {max_iter} \
--ref_data {ref_data_name} \
--run_tests {args.run_tests} \
--weights {" ".join([str(w) for w in weights])} \
""")
        output("making optimizer...")
        optimizer = Optimizer(ref_data=ref_data,
                              method=method,
                              gfn=gfn,
                              active_params=active_params,
                              max_iter=max_iter,
                              weights=weights,
                              output_func=output,
                              name=name,
                              training_set_filename=training_set_filename,
                              test_set_filename=test_set_filename,
                              validation_set_filename=validation_set_filename
                              )

        output("")

        # run optimization
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

        # run validation
        output("running validation...")
        optimizer.test_result(optimized_params)

        output(f"""
wall-clock time : {time.time() - start:6.3f} seconds

#######################
""")
