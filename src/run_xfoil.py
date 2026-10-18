import os
import subprocess
import shutil
import re
import os

from paths import get_path
from defaults import dryrun

def run_case(i, params, test=False, aerofoil_dir=None, results_solver_dir=None):
    try:
        airfoil = f"parsec_airfoil_{i}.dat"
        # Allow parent to provide paths (avoids spawn/fork issues on macOS)
        if aerofoil_dir is None:
            aerofoil_dir = get_path('aerofoils', test=test)

        # Dry-run: create a mock polar file so downstream steps can proceed
        if dryrun:
            os.makedirs(aerofoil_dir, exist_ok=True)
            polar_name = f"polar_{i}.txt"
            polar_path = os.path.join(aerofoil_dir, polar_name)
            # Minimal polar-like content that `extract_cl_values` can parse
            with open(polar_path, 'w') as f:
                f.write("-----\n")
                f.write("0.0 0.5\n")

            # If parent provided a results directory, move directly there
            if results_solver_dir:
                os.makedirs(results_solver_dir, exist_ok=True)
                dest_path = os.path.join(results_solver_dir, polar_name)
                shutil.move(polar_path, dest_path)
            else:
                base_path = get_path()
                move_results(base_path, polar_name, test=test)

            return i, True

        run_xfoil(airfoil, test=test, params=params)
        return i, True
    except Exception as e:
        print(f"Error in case {i}: {e}")
        return i, False

def run_xfoil(airfoil_file, params, test=False):
    """
    Run XFOIL on an airfoil file and save results to polar.txt.
    """

    # Set up default parameters for experiment
    re = params['solve']['Reynolds_number']
    alfas = params['solve']['Angle_of_attack']
    iterations = params['solve']['Iterations']

    base_path = get_path()
    airfoil_dir = get_path('aerofoils', test=test)
    extension = airfoil_file.split('airfoil_')[1].split('.dat')[0]
    polar_file = f"polar_{extension}.txt"
    xfoil_exe = os.path.join(get_path('xfoil'), "bin/xfoil")

    commands = f"LOAD {airfoil_file}\nPANE\n"
    commands += "OPER\n"
    commands += f"VISC {int(re)}\n"
    commands += f"ITER {iterations}\n"
    commands += "PACC\n"
    commands += f"{polar_file}\n"  # blank line after PACC
    commands += "\n"
    commands += "".join(f"ALFA {a}\n" for a in alfas)
    commands += "\n"
    commands += "PACC\n"
    commands += "QUIT\n"

    inp_path = os.path.join(airfoil_dir, "xfoil_input.inp")
    with open(inp_path, "w", newline="\r\n") as f:  # DOS line endings help
        f.write(commands)


    print("COMMANDS SENT TO XFOIL:\n", commands)
    # Run xfoil
    p = subprocess.Popen(
        [xfoil_exe],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=airfoil_dir,
        text=True
    )

    with open(inp_path, "r") as f:
        out, err = p.communicate(f.read())
    if err:
        print(err)

    print("XFOIL output:")
    print(out)
    if err:
        print("XFOIL error output:")
        print(err)
    cl = validate_xfoil_output(out)
    if cl:
        move_results(base_path, f"polar_{extension}.txt", test=test)

def move_results(base_dir,  file_name, test=False):
    dir_from = get_path(f'aerofoils', test=test)
    dir_to = get_path(f'results', test=test)
    os.makedirs(os.path.join(dir_to), exist_ok=True)
    dir_to = get_path(f'results_solver', test=test)
    os.makedirs(os.path.join(dir_to), exist_ok=True)

    source_path = os.path.join(dir_from, file_name)
    dest_dir = os.path.join(base_dir, dir_to)
    dest_path = os.path.join(dest_dir, file_name)

    if os.path.isfile(source_path):
        shutil.move(source_path, dest_path)
        print(f"Moved {file_name} from {dir_from} to {dir_to}")
    else:
        print(f"file path {source_path} does not exist")
     
def validate_xfoil_output(output_text, cl_tol=1e-3, rms_tol=1e-3):
    """
    Validate XFOIL output for convergence and physical validity.

    Parameters:
        output_text (str): full stdout from XFOIL
        cl_tol (float): tolerance for CL convergence
        rms_tol (float): threshold for residual convergence

    Returns:
        float: final CL value

    Raises:
        ValueError if invalid
    """

    # --- Extract iteration lines ---
    pattern = r"\s+\d+\s+rms:\s*([0-9.E+-]+).*?\n.*?CL\s*=\s*([0-9.+-E]+)"
    matches = re.findall(pattern, output_text)

    if not matches:
        print("No iteration data found in XFOIL output")
        return None

    # --- Parse values ---
    rms_values = [float(rms) for rms, _ in matches]
    cl_values = [float(cl) for _, cl in matches]

    final_cl = cl_values[-1]
    final_rms = rms_values[-1]

    # --- 1. Check convergence (residual) ---
    if final_rms > rms_tol:
        print(f"XFOIL did not converge (rms={final_rms:.2e})")
        return None

    # --- 2. Check CL stability ---
    if len(cl_values) >= 2:
        if abs(cl_values[-1] - cl_values[-2]) > cl_tol:
            print("CL did not stabilise")
            return None

    # --- 3. Check physical validity ---
    if final_cl < 0:
        print(f"Invalid CL value: {final_cl}")
        return None

    # --- 4. Check explicit failure messages ---
    if "VISCAL:  Convergence failed" in output_text:
        print("XFOIL viscous solver failed")
        return None

    return final_cl

def saveTimeDataGeneration(time, test=False):
    file_path = os.path.join(get_path(f'results_solver', test=test), "time.txt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        f.write(f"{time:.3f}")
    print(f"Saved {file_path} as {time}")
