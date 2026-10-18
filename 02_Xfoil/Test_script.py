import subprocess
import os

# Define the path structure
base_dir = os.path.dirname(os.path.abspath("Xfoil"))  # Base directory
xfoil_path = os.path.join(base_dir, "bin", "xfoil")  # Path to XFOIL executable
aerofoil_file = os.path.join(base_dir, "aerofoils", "naca0012.dat")  # Aerofoil file
output_file = os.path.join(base_dir, "results", "xfoil_output.txt")  # Results file

# Create the input file
input_file = 'xfoil_input.txt'
with open(input_file, "w") as f:
    #f.write(f"LOAD\n{aerofoil_file}\n")  # Load the aerofoil file
    f.write("NACA 0012\n")  # Enter operating mode
    f.write("OPER\n")  # Enter operating mode
    f.write("v\n")  # Enter Reynolds number
    f.write("1e5\n")  # Reynolds number
    f.write("alfa 10\n")  # Set angle of attack (5 degrees)
    f.write(f"PWRT\n{output_file}\n\nQUIT\n")  # Save results and quit

# Run XFOIL from the xfoil/ directory
try:
    result = subprocess.run(
        [xfoil_path],
        stdin=open(input_file),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.join(base_dir, "bin"),  # Set the working directory for XFOIL
    )
    print("XFOIL execution completed.")
    print("Output:")
    print(result.stdout.decode())  # Print XFOIL console output (optional)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Check if the XFOIL executable and paths are set correctly.")
