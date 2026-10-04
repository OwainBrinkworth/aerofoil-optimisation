import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

from run_xfoil import run_xfoil, move_results
from aerofoil_initialisation import generate_parsec_lhs
from defaults import params
from parsecairfoils import generate_foils, plot_foils, plot_airfoil
from compile_results import extract_cl_values
from random_forest_regressor import train_surrogate_model

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sample_type = params['experiment']['sampling_method']
    n_samples = params['experiment']['n_samples']
    re = params['solve']['Reynolds_number']
    alfas = params['solve']['Angle_of_attack']
    iterations = params['solve']['Iterations']
    df = pd.DataFrame()
    df["CL"] = extract_cl_values(BASE_DIR)
    bad = df[df["CL"] < 0.3]
    idx = bad.index[0]

    plot_foils(BASE_DIR, [idx])
    plot_airfoil(BASE_DIR, idx)

    # airfoil = f"parsec_airfoil_{idx}.dat"
    # run_xfoil(BASE_DIR, airfoil)
    # cl_values = extract_cl_values(BASE_DIR)

    # save_path = os.path.join(BASE_DIR, "05_Analysis", f"input_histogram_{sample_type}_{n_samples}.png")
    # plt.hist(df["CL"].dropna(), bins=50)
    # plt.xlabel("CL")
    # plt.ylabel("Frequency")
    # plt.title("CL Distribution")
    # plt.savefig(save_path, dpi=300, bbox_inches="tight")
