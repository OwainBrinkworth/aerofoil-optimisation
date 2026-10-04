# Generate and plot the contour of an airfoil 
# using the PARSEC parameterization

# Repository & documentation:
# http://github.com/dqsis/parsec-airfoils
# -------------------------------------


# Import libraries
from __future__ import division
import os

from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt

import parseccoef as pc

from defaults import params
from paths import get_path

def generate_foils(test=False, index=None):
    test_type = 'test' if test else 'experiment'
    n_samples = params[test_type]['n_samples']
    type = params[test_type]['sampling_method']
    input_directory = get_path('pre', test=test)
    output_directory = get_path('aerofoils', test=test)

    full_path = os.path.join(output_directory)
    os.makedirs(full_path, exist_ok=True)

    # Read parsec parameters (user input) & assign to array
    pp = f'parsec_init_{type}_{n_samples}.csv'
    pparray = np.genfromtxt(
        os.path.join(input_directory, pp),
        delimiter=';',
        dtype=float,
        skip_header=1
    )

    # TE & LE of airfoil (normalized, chord = 1)
    xle = 0.0
    yle = 0.0
    xte = 1.0
    yte = 0.0

    # LE radius
    aerofoils = [index] if index else range(len(pparray))

    invalid_aerofoils = []
    for i in aerofoils:
        rle = pparray[i,0]

        # Pressure (lower) surface parameters 
        x_pre = pparray[i, 1]
        y_pre = pparray[i, 2]
        d2ydx2_pre = pparray[i, 3]
        th_pre = pparray[i, 4]

        # Suction (upper) surface parameters
        x_suc = pparray[i, 5]
        y_suc = pparray[i, 6]
        d2ydx2_suc = pparray[i, 7]
        th_suc = pparray[i, 8]


        # Evaluate pressure (lower) surface coefficients
        cf_pre = pc.pcoef(xte,yte,rle,
                        x_pre,y_pre,d2ydx2_pre,th_pre,
                        'pre')

        # Evaluate suction (upper) surface coefficients
        cf_suc = pc.pcoef(xte,yte,rle,
                        x_suc,y_suc,d2ydx2_suc,th_suc,
                        'suc')


        # Evaluate pressure (lower) surface points
        xx_pre = np.linspace(xte,xle,101)
        yy_pre = (cf_pre[0]*xx_pre**(1/2) + 
                cf_pre[1]*xx_pre**(3/2) + 
                cf_pre[2]*xx_pre**(5/2) + 
                cf_pre[3]*xx_pre**(7/2) + 
                cf_pre[4]*xx_pre**(9/2) + 
                cf_pre[5]*xx_pre**(11/2)
                ) 

        # Evaluate suction (upper) surface points
        xx_suc = np.linspace(xle,xte,101)
        yy_suc = (cf_suc[0]*xx_suc**(1/2) + 
                cf_suc[1]*xx_suc**(3/2) + 
                cf_suc[2]*xx_suc**(5/2) + 
                cf_suc[3]*xx_suc**(7/2) + 
                cf_suc[4]*xx_suc**(9/2) + 
                cf_suc[5]*xx_suc**(11/2)
                )
        
        # Return surfaces for plotting
        if index:
               return xx_suc, yy_suc, xx_pre, yy_pre

        # Use parsecexport to save coordinate file
        fpath = os.path.join(full_path, f'parsec_airfoil_{i}.dat')
        # with ... as ... only opens the file for the block it executes, then closes it
        plain_coords = ppointsplain(i, cf_pre, cf_suc, 241, xte=xte)
        if plain_coords:
            with open(fpath, 'w') as f:
                    f.write(plain_coords)
        else:
             invalid_aerofoils.append(i)
    return invalid_aerofoils

def plot_foils(aerofoil_indices):
        n_samples = params['sample']['n_samples']
        type = params['sample']['sampling_method']
        input_directory = get_path('pre_experiment')
        pp = f'parsec_init_{type}_{n_samples}.csv'

        for aerofoil_index in aerofoil_indices:
                xx_suc, yy_suc, xx_pre, yy_pre = generate_foils(index=aerofoil_index)
                # Plot airfoil contour
                plt.figure()

                plt.plot(xx_suc,yy_suc,'r',
                        xx_pre,yy_pre,'b', linewidth=2)

                plt.grid(True)
                plt.xlim([0,1])
                #plt.yticks([])
                plt.xticks(np.arange(0,1.1,0.1))
                plt.gca().axis('equal')

                # Some magic with strings
                with open(os.path.join(input_directory, pp), 'r') as f:
                        lines = f.readlines()

                # First line = headers
                parnames = lines[0].strip().split(';')

                # Data starts from line 1 → so offset by +1
                parvals = lines[aerofoil_index + 1].strip().split(';')

                parnv = [f"{n}={float(v):.3f}" for n, v in zip(parnames, parvals)]
                parnv = [', '.join(parnv[:5]), ', '.join(parnv[5:])]

                plt.title("PARSEC airfoil with parameters:\n{}\n{}"
                        .format(parnv[0], parnv[1]))
                # Make room for title automatically
                plt.tight_layout()

                plt.savefig(os.path.join(input_directory, f'parsec_airfoil_{type}_{n_samples}_{aerofoil_index}.png'))

def plot_airfoil(index, coords=np.array([]), valid=True):
    """
    Plot an airfoil from a .dat file and save the figure.

    Parameters:
        dat_path (str): path to .dat file
        save_dir (str): directory to save plot
        filename (str): output image filename
    """

    # --- Read coordinates ---
    n_samples = params['experiment']['n_samples']
    type = params['experiment']['sampling_method']
    if coords.any():
        x = coords[:, 0]
        y = coords[:, 1]
    else:
        input_directory = get_path('aerofoils')
        pp = f'parsec_airfoil_{index}.dat'
        x, y = [], []
        dat_path = os.path.join(input_directory, pp)

        with open(dat_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        xi, yi = float(parts[0]), float(parts[1])
                        x.append(xi)
                        y.append(yi)
                    except:
                        continue  # skip invalid lines

    # --- Plot ---
    plt.figure()
    plt.plot(x, y)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Airfoil geometry")

    # --- Ensure save directory exists ---
    save_dir = get_path('aerofoils')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'parsec_airfoil_{type}_{n_samples}_{index}_dat_valid_{valid}.png')

    # --- Save ---
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Airfoil plot saved to: {save_path}")

def ppointsplain(index, *args, **kwargs):
    '''Alias for ppoints that returns a string with plain data format'''
    coords = ppoints(*args, **kwargs)

    # Check the aerofoils
    if validate_aerofoil(coords):
        # Iterate over coordinates, making a list of strings
        coordstrlist = ["{:.6f} {:.6f}".format(coord[0], coord[1])
                        for coord in coords]
        plot_airfoil(index, coords=coords, valid=True)
        # Now join these strings with linebreaks in between
        return '\n'.join(coordstrlist)
    else:
        plot_airfoil(index, coords=coords, valid=False)
        return ''

def validate_aerofoil(coord):
    x = coord[:, 0]
    y = coord[:, 1]
    le_idx = np.argmin(x)

    upper = coord[:le_idx+1]
    lower = coord[le_idx:]
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]

    x_common = np.linspace(0.0, 1.0, 500)

    upper_interp = interp1d(upper[:, 0], upper[:, 1], bounds_error=False, fill_value='extrapolate')
    lower_interp = interp1d(lower[:, 0], lower[:, 1], bounds_error=False, fill_value='extrapolate')

    y_upper = upper_interp(x_common)
    y_lower = lower_interp(x_common)

    thickness = y_upper - y_lower

    t_max_idx = np.argmax(thickness)
    t_max = thickness[t_max_idx]
    x_t_max = x_common[t_max_idx]

    # Check crossing
    if np.any(thickness <= 0.00):
        return False

    # Check maximum thickness bounds
    if t_max < 0.04 or t_max > 0.25:
        return False

    # Check maximum thickness location
    if x_t_max < 0.1 or x_t_max > 0.6:
        return False

    return True


def ppoints(cf_pre, cf_suc, npts=121, xte=1.0):
    '''
    Takes PARSEC coefficients, number of points, and returns list of
    [x,y] coordinates starting at trailing edge pressure side.
    Assumes trailing edge x position is 1.0 if not specified.
    Returns 121 points if 'npts' keyword argument not specified.
    '''
    # Using cosine spacing to concentrate points near TE and LE,
    # see http://airfoiltools.com/airfoil/naca4digit
    xpts = (1 - np.cos(np.linspace(0, 1, int(np.ceil(npts/2)))*np.pi)) / 2
    # Take TE x-position into account
    xpts *= xte

    # Powers to raise coefficients to
    pwrs = (1/2, 3/2, 5/2, 7/2, 9/2, 11/2)
    # Make [[1,1,1,1],[2,2,2,2],...] style array
    xptsgrid = np.meshgrid(np.arange(len(pwrs)), xpts)[1]
    # Evaluate points with concise matrix calculations. One x-coordinate is
    # evaluated for every row in xptsgrid
    evalpts = lambda cf: np.sum(cf*xptsgrid**pwrs, axis=1)
    # Move into proper order: start at TE, over bottom, then top
    # Avoid leading edge pt (0,0) being included twice by slicing [1:]
    ycoords = np.append(evalpts(cf_pre)[::-1], evalpts(cf_suc)[1:])
    xcoords = np.append(xpts[::-1], xpts[1:])
    # Return 2D list of coordinates [[x,y],[x,y],...] by transposing .T
    return np.array((xcoords, ycoords)).T


def _example():
    '''Runs some examples. Underscore means it's a local function.'''
    # Sample coefficients
    rle = .01
    x_pre = .45
    y_pre = -0.006
    d2ydx2_pre = -.2
    th_pre = 2
    x_suc = .35
    y_suc = .055
    d2ydx2_suc = -.35
    th_suc = -10

    # Trailing edge x and y position
    xte = .9
    yte = -0.05

    # Evaluate pressure (lower) surface coefficients
    cf_pre = pc.pcoef(xte, yte, rle,
                      x_pre, y_pre, d2ydx2_pre, th_pre,
                      'pre')
    # Evaluate suction (upper) surface coefficients
    cf_suc = pc.pcoef(xte, yte, rle,
                      x_suc, y_suc, d2ydx2_suc, th_suc,
                      'suc')

    # Get and print plain list of coordinates
    print(ppointsplain(cf_pre, cf_suc, 121, xte=xte))
    # Get list of coordinates
    pts = ppoints(cf_pre, cf_suc, 121, xte=xte)
    # Plot this airfoil, transposing pts to get list of x's and list of y's
    import matplotlib.pyplot as plt
    plt.plot(pts.T[0], pts.T[1], 'o--')
    plt.gca().axis('equal')
    plt.show()

# If this file is run, execute example
if __name__ == "__main__":
    _example()
