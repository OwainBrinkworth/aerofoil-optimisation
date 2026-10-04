import time

from run_xfoil import run_case, saveTimeDataGeneration
from aerofoil_initialisation import generate_parsec_lhs, check_sample_exists, getAerofoilsAsDataframe
from defaults import params
from paths import get_path
from parsecairfoils import generate_foils, plot_airfoil
from compile_results import extract_cl_values
from random_forest_regressor import train_surrogate_model
from multiprocessing import Pool, cpu_count
from analysis import analysis
from functools import partial

if __name__ == "__main__":
    def processAerofoils(test=False):
        n_samples = params['test']['n_samples'] if test else params['experiment']['n_samples']
        start_time = time.time()
        results_df = generate_parsec_lhs(test)
        invalid_aerofoils = generate_foils(test)
        for i in range(n_samples):
            plot_airfoil(i)

        # Use slightly fewer than max cores (keeps system stable)
        n_cores = max(1, cpu_count() - 1)

        # Compute paths here and pass into worker to avoid multiprocessing 'spawn' import issues
        aerofoil_dir = get_path('aerofoils', test=test)
        results_solver_dir = get_path('results_solver', test=test)

        run_case_with_flag = partial(
            run_case,
            test=test,
            params=params,
            aerofoil_dir=aerofoil_dir,
            results_solver_dir=results_solver_dir,
        )

        with Pool(n_cores) as pool:
            results = pool.map(run_case_with_flag, range(n_samples))
        end_time = time.time()
        saveTimeDataGeneration(end_time-start_time, test=test)
        return results_df

    # Let's Go!
    sampling_exists = check_sample_exists(test=False)
    tests_exist = check_sample_exists(test=True)
    if sampling_exists:
        results_df = getAerofoilsAsDataframe(test=False)
        # Attempt to populate CL values from any existing solver output.
        results_df = extract_cl_values(results_df)
        # If no valid CLs found, run the solver (or dryrun) to generate results
        if results_df["CL"].dropna().empty:
            print("No solver results found for experiment sampling; running cases now...")
            results_df = processAerofoils(test=False)
    else:
        results_df = processAerofoils(test=False)
    
    if not tests_exist:
        processAerofoils(test=True)

    results_df = extract_cl_values(results_df)
    # If there are no valid CL values after extraction, abort training gracefully
    if results_df["CL"].dropna().empty:
        print("No CL values available after running solver — skipping model training.")
    else:
        model, X_test, y_test, training_time = train_surrogate_model(results_df)
        analysis(model, results_df, training_time, X_test=X_test, y_test=y_test)

