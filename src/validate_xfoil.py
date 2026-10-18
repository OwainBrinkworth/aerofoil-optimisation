"""
Run one-off XFOIL validation cases and produce correlation plots.

This script has two uses:

    1. run_single_case(...) — run one XFOIL case (any Re/Mach/angle range/
       solver settings you like) and, if experimental data is available,
       plot it against that data with an RMSE. Useful for quickly checking
       a specific setup.

    2. main() — reproduces the two standard validation studies used in the
       project report:
         - Ncrit sensitivity: compares XFOIL's NACA 0012 lift curve against
           NASA/Ladson wind-tunnel data for several Ncrit values.
         - Iteration sensitivity: plots residual convergence against
           iteration count, to help choose a sensible max iteration limit
           (no experimental comparison needed for this one).

Run this file directly (`python validate_xfoil.py`) to execute both studies
and save their plots/CSVs under 03_Results/Validation/.
"""

import os

from xfoil_validation import (
    EXPERIMENTAL_DATA,
    REYNOLDS_NUMBER,
    MACH_NUMBER,
    AOA,
    BASE_ITERATIONS,
    BASE_NCRIT,
    ITERATION_VALUES,
    NCRIT_VALUES,
    XFOIL_EXE,
    run_xfoil,
    read_xfoil_polar,
    diagnose_polar,
    calculate_errors,
    plot_lift_curves,
    plot_iteration_residuals,
    run_study,
)
from paths import get_path


def run_single_case(
    case_name="single_case",
    aoa=AOA,
    re=REYNOLDS_NUMBER,
    mach=MACH_NUMBER,
    iterations=BASE_ITERATIONS,
    ncrit=BASE_NCRIT,
    experimental_data=None,
    output_dir=None,
    show=False,
):
    """
    Run a single one-off XFOIL case and plot the resulting lift curve.

    Parameters:
        case_name (str): label used for output files and plot legend.
        aoa (array-like): angles of attack to sweep (degrees).
        re, mach (float): flow conditions.
        iterations, ncrit (int): solver settings.
        experimental_data (dict or None): optional {"alpha": [...], "cl": [...]}
            to overlay and compute RMSE against. Defaults to the NASA/Ladson
            data for this Reynolds number if available, otherwise no
            experimental comparison is made.
        output_dir (str or None): where to save results. Defaults to a
            "single_case" folder under the validation results directory.
        show (bool): display the plot interactively as well as saving it.

    Returns:
        dict: the XFOIL polar data, error metrics (if compared) and file
            paths for this case.
    """
    if output_dir is None:
        output_dir = get_path("results_validation", description="single_case")
    os.makedirs(output_dir, exist_ok=True)

    if experimental_data is None:
        experimental_data = EXPERIMENTAL_DATA.get(re)

    run = run_xfoil(case_name=case_name, aoa=aoa, re=re, mach=mach, iterations=iterations, ncrit=ncrit, output_dir=output_dir)
    xfoil_data = read_xfoil_polar(run["polar_path"])
    diagnostics = diagnose_polar(xfoil_data, aoa, run["stdout"])

    result = {"case": case_name, "iterations": iterations, "ncrit": ncrit, "run": run, "xfoil": xfoil_data, "diagnostics": diagnostics}

    if experimental_data is not None:
        error = calculate_errors(xfoil_data, experimental_data, aoa)
        result["error"] = error
        print(f"{case_name}: RMSE={error['rmse']:.5f} against {error['n_points']} experimental points")

        plot_lift_curves(
            results=[result], experimental_data=experimental_data, requested_aoa=aoa,
            title=f"{case_name}: XFOIL vs experimental data (Re={re:.1e})",
            plot_path=os.path.join(output_dir, f"{case_name}_cl_vs_alpha.png"),
            show=show,
        )
    else:
        print(f"{case_name}: no experimental data available for Re={re:.1e}; skipping comparison plot")

    return result


def main():
    """Run the Ncrit and iteration validation studies used in the report."""
    if not os.path.isfile(XFOIL_EXE):
        raise FileNotFoundError(f"XFOIL executable not found: {XFOIL_EXE}")

    experimental_data = EXPERIMENTAL_DATA[REYNOLDS_NUMBER]

    # --------------------------------------------------------
    # 1. Ncrit sensitivity vs experimental data.
    # Hold the iteration limit constant; vary the transition criterion.
    # --------------------------------------------------------
    ncrit_cases = [
        {"case": f"ncrit_{n}", "iterations": BASE_ITERATIONS, "ncrit": n}
        for n in NCRIT_VALUES
    ]
    run_study(
        study_name="ncrit",
        description="ncrit_sensitivity",
        cases=ncrit_cases,
        experimental_data=experimental_data,
    )

    # --------------------------------------------------------
    # 2. Iteration sensitivity: residual convergence only, no experimental
    # comparison. Used to pick a sensible max iteration count.
    # --------------------------------------------------------
    iteration_cases = [
        {"case": f"iterations_{n}", "iterations": n, "ncrit": BASE_NCRIT}
        for n in ITERATION_VALUES
    ]
    iteration_output_dir = get_path("results_validation", description="iteration_sensitivity")
    plot_iteration_residuals(cases=iteration_cases, output_dir=iteration_output_dir)


if __name__ == "__main__":
    main()
