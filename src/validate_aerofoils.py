"""
Validate the shape of specific PARSEC aerofoils.

This is a quick way to check that a chosen set of PARSEC parameters
produces a sensible aerofoil shape *before* spending time running it
through XFOIL or including it in a training/test dataset.

For each aerofoil defined in AEROFOILS_TO_CHECK below, this script:
    1. Generates the aerofoil's surface coordinates from its PARSEC
       parameters.
    2. Runs the same geometry validity checks used elsewhere in the
       pipeline (no self-intersection, sensible thickness, etc.).
    3. Saves a diagnostic plot (contour, thickness distribution, thickness
       gradient, surface gradients) to the results/validation folder.
    4. Prints a plain-English pass/fail summary.

To check your own aerofoil, add an entry to AEROFOILS_TO_CHECK with its
PARSEC parameters (see defaults.py's `param_bounds` for the valid range of
each parameter).
"""

import os

from paths import get_path
from parsecairfoils import plot_aerofoil_diagnostics, _validation_reasons


# Each entry is a (name, parsec_params) pair. The default entry here
# reproduces the example aerofoil previously hard-coded in
# parsecairfoils.py's _example() function.
AEROFOILS_TO_CHECK = [
    (
        "example_aerofoil",
        dict(
            R_LE=0.01,
            x_PRE=0.45,
            y_PRE=0.060,
            d2ydx2_PRE=-0.2,
            th_PRE=2.0,
            x_SUC=0.35,
            y_SUC=0.025,
            d2ydx2_SUC=-0.35,
            th_SUC=-10.0,
        ),
    ),
]


def validate_aerofoils(aerofoils_to_check=AEROFOILS_TO_CHECK, show=False):
    """
    Generate and validate each aerofoil in aerofoils_to_check.

    Parameters:
        aerofoils_to_check (list): list of (name, parsec_params) pairs.
        show (bool): if True, also display each plot interactively.

    Returns:
        dict: name -> True/False validity result, for easy scripting use.
    """
    output_dir = get_path('results_validation', description='aerofoil_shape_checks')
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for name, parsec_params in aerofoils_to_check:
        save_path = os.path.join(output_dir, f"{name}_diagnostics.png")
        _, details = plot_aerofoil_diagnostics(
            parsec_params, show=show, save_path=save_path,
        )
        results[name] = details['valid']

        status = "PASSED" if details['valid'] else "FAILED"
        print(f"[{status}] {name}: max thickness {details['t_max']:.4f} at x/c={details['x_t_max']:.4f}")
        if not details['valid']:
            for reason in _validation_reasons(details):
                print(f"    - {reason}")

    return results


if __name__ == "__main__":
    validate_aerofoils()
