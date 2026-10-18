import subprocess
import os
import numpy as np
from scipy.optimize import minimize


# Define a function to run XFOIL and get performance data
def run_xfoil(aerofoil_name, alpha=5.0, camb=0.0, thick=0.0):
    """
    Run XFOIL with specified aerofoil and parameters.
    :param aerofoil_name: Name of the aerofoil file (e.g., 'naca0012.dat')
    :param alpha: Angle of attack (degrees)
    :param camb: Camber modification
    :param thick: Thickness modification
    :return: Lift-to-drag ratio (L/D)
    """
    # Create input file for XFOIL
    input_file = 'xfoil_input.txt'
    output_file = 'xfoil_output.txt'

    with open(input_file, 'w') as f:
        f.write(f"LOAD\n{aerofoil_name}\n")
        if camb > 0.0 or thick > 0.0:
            f.write(f"MDES\n{camb}\n{thick}\n\n")
            f.write("EXEC\n\n")
        f.write("OPER\n")
        f.write(f"alfa {alpha}\n")
        f.write(f"PWRT\n{output_file}\n\nQUIT\n")

    # Run XFOIL as a subprocess
    process = subprocess.run(['xfoil'], stdin=open(input_file), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read the results from the output file
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
            # Extract L/D ratio from the last line
            for line in lines:
                if 'CL/CD' in line:
                    l_d_ratio = float(line.split()[-1])  # Extract the L/D ratio
                    return l_d_ratio
    except FileNotFoundError:
        print("Error: XFOIL output file not found.")
        return None
    finally:
        # Clean up temporary files
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)


# Define the objective function for optimization
def objective(params):
    """
    Objective function for optimization. The goal is to maximize L/D.
    :param params: [camber, thickness]
    :return: Negative L/D (for minimization)
    """
    camb, thick = params
    l_d_ratio = run_xfoil("naca0012.dat", alpha=5.0, camb=camb, thick=thick)
    if l_d_ratio is None:
        return 1e6  # Return a high penalty for invalid results
    return -l_d_ratio  # Negative because we want to maximize L/D


# Optimization setup
initial_guess = [0.02, 0.12]  # Initial guesses for camber and thickness
bounds = [(0.0, 0.1), (0.05, 0.2)]  # Bounds for camber and thickness

# Perform the optimization
result = minimize(objective, initial_guess, bounds=bounds, method='SLSQP')

# Display the optimal parameters and performance
optimal_camber, optimal_thickness = result.x
optimal_l_d = -result.fun

print(f"Optimal camber: {optimal_camber:.4f}")
print(f"Optimal thickness: {optimal_thickness:.4f}")
print(f"Optimal lift-to-drag ratio (L/D): {optimal_l_d:.2f}")
