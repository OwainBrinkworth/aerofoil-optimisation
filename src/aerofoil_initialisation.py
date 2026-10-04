import numpy as np
import pandas as pd
import os

from scipy.stats import qmc
from defaults import params, param_bounds
from paths import get_path

def check_sample_exists(test):
    directory = get_sample_directory(get_path('pre', test=test), test=test)
    return os.path.isfile(directory)

def get_sample_directory(base_path, test=False):
    test_type = 'test' if test else 'experiment'
    n_samples = params[test_type]['n_samples']
    method = params[test_type]['sampling_method']
    return os.path.join(base_path, f"parsec_init_{method}_{n_samples}.csv")

def getAerofoilsAsDataframe(test=False):
        directory = get_path('pre', test=test)
        csv_path = get_sample_directory(directory, test=test)
        # --- Load input parameters ---
        return pd.read_csv(csv_path, sep=";")

def generate_parsec_lhs(test=False):
    """
    Generate PARSEC samples for EXTRAPOLATION only.

    Uses expanded bounds and removes samples inside original bounds.

    Ensures ~n_samples output via oversampling.
    """

    test_type = 'test' if test else 'experiment'
    base_path = get_path('pre', test=test)

    n_samples = params[test_type]['n_samples']
    method = params[test_type]['sampling_method']
    if test_type == 'test' and not params['test']['interpolation']:
        expansion_ratio = params['test']['expansion_ratio']
    else:
         expansion_ratio = 1

    param_names = list(param_bounds.keys())
    bounds = np.array(list(param_bounds.values()))

    dim = len(param_names)

    # =========================
    # 1. Define expanded bounds
    # =========================
    l_bounds = bounds[:, 0]
    u_bounds = bounds[:, 1]

    centre = (l_bounds + u_bounds) / 2
    half_range = (u_bounds - l_bounds) / 2

    expanded_half_range = half_range * expansion_ratio

    l_exp = centre - expanded_half_range
    u_exp = centre + expanded_half_range

    # =========================
    # 2. Oversample (important)
    # =========================
    if expansion_ratio == 1:
         oversample_factor = 1
    else:
        oversample_factor = 1 / (1 - expansion_ratio ** (-9))  # number of parsec parameters
    n_generate = int(n_samples * oversample_factor)

    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=n_generate)

    scaled_sample = qmc.scale(sample, l_exp, u_exp)

    df = pd.DataFrame(scaled_sample, columns=param_names)

    if expansion_ratio != 1:
        # =========================
        # 3. Remove inner points
        # =========================
        mask = np.zeros(len(df), dtype=bool)

        for i, param in enumerate(param_names):
            lower, upper = l_bounds[i], u_bounds[i]

            outside = (df[param] < lower) | (df[param] > upper)
            mask = mask | outside  # ANY dimension outside

        df_extrap = df[mask]

        # =========================
        # 4. Ensure enough samples
        # =========================
        if len(df_extrap) < n_samples:
            print(f"Warning: only {len(df_extrap)} extrapolation samples found, increasing oversampling")
        
        # Truncate to desired size
        df_extrap = df_extrap.iloc[:n_samples]
    else:
         df_extrap = df

    # =========================
    # 5. Save
    # =========================
    os.makedirs(os.path.join(base_path), exist_ok=True)
    output_file = os.path.join(
        base_path,
        f"parsec_init_{method}_{n_samples}.csv"
    )

    df_extrap.to_csv(output_file, sep=";", index=False)

    print(f"Generated {len(df_extrap)} extrapolation aerofoils → {output_file}")

    return df_extrap
