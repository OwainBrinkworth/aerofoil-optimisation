"""
Main entry point for the aerofoil surrogate-modelling experiment.

Running this script performs the full pipeline, using the parameters set in
defaults.py:

    1. Sample PARSEC parameters (Latin Hypercube Sampling) for both the
       training ("experiment") set and the held-out test set.
    2. Generate aerofoil geometry (.dat coordinate files) for each sample.
    3. Run XFOIL on each aerofoil to get its lift coefficient (CL).
    4. Train a surrogate model (e.g. random forest) to predict CL directly
       from the PARSEC parameters, skipping XFOIL entirely.
    5. Analyse how well the surrogate model performs against the XFOIL
       results and produce plots/metrics.

If a sampling run has already been completed (its results exist on disk),
that step is skipped and the existing results are reused instead of being
regenerated.
"""

import time

from run_xfoil import run_case, saveTimeDataGeneration
from aerofoil_initialisation import generate_parsec_lhs, check_sample_exists, getAerofoilsAsDataframe
from defaults import params, dryrun
from paths import get_path
from parsecairfoils import generate_foils
from compile_results import extract_cl_values
from models import train_surrogate_model
from analysis import analysis


def process_aerofoils(test=False):
    """
    Generate a full batch of aerofoils and run XFOIL on each one.

    Parameters:
        test (bool): if True, use the held-out test sampling settings from
            defaults.py instead of the main experiment settings.

    Returns:
        pandas.DataFrame: one row per aerofoil, with its PARSEC parameters
            (CL is filled in later by extract_cl_values once XFOIL has run).
    """
    n_samples = params['test']['n_samples'] if test else params['experiment']['n_samples']
    start_time = time.time()

    generate_parsec_lhs(test)
    generate_foils(test)
    results_df = getAerofoilsAsDataframe(test=test)

    # Resolve these paths once up front (rather than inside run_case) so the
    # same values are used for every aerofoil in the batch.
    aerofoil_dir = get_path('aerofoils', test=test)
    results_solver_dir = get_path('results_solver', test=test)

    # Run XFOIL on each aerofoil one at a time. This is run sequentially
    # (not in parallel) to keep XFOIL's file I/O simple and predictable.
    for i in range(n_samples):
        run_case(i, params=params, test=test, aerofoil_dir=aerofoil_dir, results_solver_dir=results_solver_dir)

    end_time = time.time()
    saveTimeDataGeneration(end_time - start_time, test=test)
    return results_df


if __name__ == "__main__":
    # --- 1. Training ("experiment") data set ---
    # Reuse an existing sample/result set if one is already on disk;
    # otherwise generate aerofoils and run XFOIL from scratch.
    sampling_exists = check_sample_exists(test=False)
    if sampling_exists and not dryrun:
        results_df = getAerofoilsAsDataframe(test=False)
        results_df = extract_cl_values(results_df)
        if results_df["CL"].dropna().empty:
            print("No solver results found for experiment sampling; running cases now...")
            results_df = process_aerofoils(test=False)
    else:
        results_df = process_aerofoils(test=False)

    # --- 2. Held-out test data set ---
    tests_exist = check_sample_exists(test=True)
    if not tests_exist:
        process_aerofoils(test=True)

    # --- 3. Train and analyse the surrogate model ---
    results_df = extract_cl_values(results_df)
    if results_df["CL"].dropna().empty:
        print("No CL values available after running solver — skipping model training.")
    else:
        model, X_test, y_test, training_time = train_surrogate_model(results_df)
        analysis(model, results_df, training_time, X_test=X_test, y_test=y_test)

