import os

from defaults import params, dryrun

def get_path(key=None, test=False, description=''):
    """
    Returns structured project paths based on a keyword.

    Parameters:
        key (str): which path to return
        params (dict): parameter dictionary
        base_path (str): project base directory
        run_id (str): run identifier (e.g. '001')
        dryrun (bool): whether this is a dry run

    Returns:
        str: full path
    """
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    run_type = 'test' if test else 'experiment'

    # --- Extract parameters ---
    sampling_method = params[run_type]['sampling_method']
    n_samples = params[run_type]['n_samples']
    Re = int(params['solve']['Reynolds_number'])
    aoa = params['solve']['Angle_of_attack']

    # Format AoA string
    aoa_str = "_".join(str(a) for a in aoa)

    # Experiment type
    experiment_type = "Dryrun" if dryrun else "Experiments"
    test_type = "Interpolation" if params['test']['interpolation'] else "Extrapolation"

    if not key:
        key = 'base'

    # --- Define paths ---
    if test and test_type == "Interpolation":
        paths = {
            "base": base_path,

            "pre": os.path.join(
                base_path,
                "00_Pre",
                "Tests",
                test_type,
                ),

            "aerofoils": os.path.join(
                base_path,
                "01_Aerofoils",
                "Tests",
                test_type,
                f"{sampling_method}_{n_samples}",
            ),

            "xfoil": os.path.join(base_path, "02_Xfoil"),

            "results": os.path.join(
                base_path,
                "03_Results",
                "Tests",
                test_type,
                f"{sampling_method}_{n_samples}",
            ),

            "results_solver": os.path.join(
                base_path,
                "03_Results",
                "Tests",
                test_type,
                f"{sampling_method}_{n_samples}",
                f"{Re}_{aoa_str}",
            ),

            "results_validation": os.path.join(
                base_path,
                "03_Results",
                "Validation",
                description,
            ),

            "models": os.path.join(base_path, "04_Models"),

            "analysis": os.path.join(
                base_path,
                "05_Analysis",
                experiment_type,
            ),

            "sampling_time": os.path.join(
                base_path,
                "03_Results",
                "Tests",
                test_type,
                f"{sampling_method}_{n_samples}",
                f"{Re}_{aoa_str}",
                "time.txt",
            ),
        }
    elif test and test_type == "Extrapolation":
        expansion_ratio = f"{params['test']['expansion_ratio']}".replace('.', '_')
        paths = {
            "base": base_path,

            "pre": os.path.join(
                base_path,
                "00_Pre",
                "Tests",
                test_type,
                expansion_ratio,
                ),

            "aerofoils": os.path.join(
                base_path,
                "01_Aerofoils",
                "Tests",
                test_type,
                expansion_ratio,
                f"{sampling_method}_{n_samples}",
            ),

            "xfoil": os.path.join(base_path, "02_Xfoil"),

            "results": os.path.join(
                base_path,
                "03_Results",
                "Tests",
                test_type,
                expansion_ratio,
                f"{sampling_method}_{n_samples}",
            ),

            "results_solver": os.path.join(
                base_path,
                "03_Results",
                "Tests",
                test_type,
                expansion_ratio,
                f"{sampling_method}_{n_samples}",
                f"{Re}_{aoa_str}",
            ),

            "results_validation": os.path.join(
                base_path,
                "03_Results",
                "Validation",
                description,
            ),

            "models": os.path.join(base_path, "04_Models"),

            "analysis": os.path.join(
                base_path,
                "05_Analysis",
                experiment_type,
            ),

            "sampling_time": os.path.join(
                base_path,
                "03_Results",
                "Tests",
                test_type,
                expansion_ratio,
                f"{sampling_method}_{n_samples}",
                f"{Re}_{aoa_str}",
                "time.txt",
            ),
        }
    else:
        paths = {
            "base": base_path,

            "pre": os.path.join(base_path, "00_Pre", "Experiments"),

            "aerofoils": os.path.join(
                base_path,
                "01_Aerofoils",
                "Experiments",
                f"{sampling_method}_{n_samples}",
            ),

            "xfoil": os.path.join(base_path, "02_Xfoil"),

            "results": os.path.join(
                base_path,
                "03_Results",
                "Experiments",
                f"{sampling_method}_{n_samples}",
            ),

            "results_solver": os.path.join(
                base_path,
                "03_Results",
                "Experiments",
                f"{sampling_method}_{n_samples}",
                f"{Re}_{aoa_str}",
            ),

            "results_validation": os.path.join(
                base_path,
                "03_Results",
                "Validation",
                description,
            ),

            "models": os.path.join(base_path, "04_Models"),

            "analysis": os.path.join(
                base_path,
                "05_Analysis",
                experiment_type,
            ),

            "sampling_time": os.path.join(
                base_path,
                "03_Results",
                "Experiments",
                f"{sampling_method}_{n_samples}",
                f"{Re}_{aoa_str}",
                "time.txt",
            ),
        }

    if key not in paths:
        raise ValueError(f"Invalid path key: {key}")

    return paths[key]
