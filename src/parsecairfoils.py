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

from defaults import params, param_bounds, replacement
from paths import get_path


def _sample_to_coords(sample):
    """Convert a PARSEC parameter row into a coordinate array."""
    rle = float(sample[0])

    x_pre = float(sample[1])
    y_pre = float(sample[2])
    d2ydx2_pre = float(sample[3])
    th_pre = float(sample[4])

    x_suc = float(sample[5])
    y_suc = float(sample[6])
    d2ydx2_suc = float(sample[7])
    th_suc = float(sample[8])

    cf_pre = pc.pcoef(1.0, 0.0, rle, x_pre, y_pre, d2ydx2_pre, th_pre, 'pre')
    cf_suc = pc.pcoef(1.0, 0.0, rle, x_suc, y_suc, d2ydx2_suc, th_suc, 'suc')
    return ppoints(cf_pre, cf_suc, 241, xte=1.0)


def _random_parsec_sample(rng=None):
    """Draw a new PARSEC sample within the valid bounds."""
    if rng is None:
        rng = np.random.default_rng()
    lower = np.array([bounds[0] for bounds in param_bounds.values()], dtype=float)
    upper = np.array([bounds[1] for bounds in param_bounds.values()], dtype=float)
    return rng.uniform(lower, upper)


def _validation_reasons(details):
    """Return a human-readable list of reasons why an aerofoil was rejected."""
    if not isinstance(details, dict):
        return ['invalid geometry']

    reasons = []
    if not np.isfinite(details.get('thickness', [])).all():
        reasons.append('non-finite thickness profile')
    if np.any(np.asarray(details.get('thickness', []), dtype=float) <= 0.0):
        reasons.append('thickness <= 0 at some x')

    t_max = float(details.get('t_max', 0.0))
    x_t_max = float(details.get('x_t_max', 0.0))
    if t_max < 0.04 or t_max > 0.25:
        reasons.append(f'max thickness {t_max:.4f} outside [0.04, 0.25]')
    if x_t_max < 0.1 or x_t_max > 0.6:
        reasons.append(f'max thickness x-position {x_t_max:.4f} outside [0.10, 0.60]')

    if not reasons:
        reasons.append('failed geometry validity check')
    return reasons


def generate_foils(test=False, index=None):
    test_type = 'test' if test else 'experiment'
    n_samples = params[test_type]['n_samples']
    type_name = params[test_type]['sampling_method']
    input_directory = get_path('pre', test=test)
    output_directory = get_path('aerofoils', test=test)

    full_path = os.path.join(output_directory)
    os.makedirs(full_path, exist_ok=True)

    pp = f'parsec_init_{type_name}_{n_samples}.csv'
    pparray = np.genfromtxt(
        os.path.join(input_directory, pp),
        delimiter=';',
        dtype=float,
        skip_header=1,
        ndmin=2,
    )

    if index is not None:
        if index < 0 or index >= len(pparray):
            raise IndexError(f'Aerofoil index {index} is outside the {len(pparray)} samples in {pp}.')
        sampled = pparray[index]
        coords = _sample_to_coords(sampled)
        return coords[:, 0], coords[:, 1], coords[:, 0], coords[:, 1]

    rng = np.random.default_rng() if replacement else None
    max_replacement_attempts = max(50, n_samples * 100) if replacement else 1
    csv_path = os.path.join(input_directory, pp)
    with open(csv_path, 'r') as csv_file:
        csv_header = csv_file.readline().strip()

    for sample_index in range(n_samples):
        candidate = pparray[sample_index].copy()
        attempts = 0
        final_coords = None
        while attempts < max_replacement_attempts:
            coords = _sample_to_coords(candidate)
            valid, details = validate_aerofoil(coords, return_details=True)
            reasons = _validation_reasons(details)
            plot_airfoil(
                sample_index,
                coords=coords,
                valid=valid,
                validity_details=details,
                attempt_number=attempts + 1,
                rejection_reasons=reasons,
            )

            if valid:
                final_coords = coords
                break

            print(f"Invalid aerofoil for sample {sample_index}, attempt {attempts + 1}: {', '.join(reasons)}")
            attempts += 1
            if replacement:
                candidate = _random_parsec_sample(rng)
            else:
                final_coords = coords
                break
        else:
            raise RuntimeError(f"Unable to generate a valid replacement for sample {sample_index} after {max_replacement_attempts} attempts.")

        if final_coords is None:
            raise RuntimeError(f"No generated aerofoil for sample {sample_index}.")

        if replacement and attempts > 0:
            pparray[sample_index] = candidate
            np.savetxt(
                csv_path,
                pparray,
                delimiter=';',
                fmt='%.17g',
                header=csv_header,
                comments='',
            )

        fpath = os.path.join(full_path, f'parsec_airfoil_{sample_index}.dat')
        plain_coords = '\n'.join(f"{coord[0]:.6f} {coord[1]:.6f}" for coord in final_coords)
        with open(fpath, 'w') as f:
            f.write(plain_coords)

    return []


def coords_to_string(coords, index, valid=True, attempt_number=None):
    """Return a plain coordinate text block and save a validation plot for the geometry."""
    valid, details = validate_aerofoil(coords, return_details=True)
    reasons = _validation_reasons(details)
    plot_airfoil(
        index,
        coords=coords,
        valid=valid,
        validity_details=details,
        attempt_number=attempt_number,
        rejection_reasons=reasons,
    )
    if not valid:
        return ''
    return '\n'.join(f"{coord[0]:.6f} {coord[1]:.6f}" for coord in coords)


def plot_foils(aerofoil_indices):
    n_samples = params['sample']['n_samples']
    type_name = params['sample']['sampling_method']
    input_directory = get_path('pre_experiment')
    pp = f'parsec_init_{type_name}_{n_samples}.csv'

    for aerofoil_index in aerofoil_indices:
        xx_suc, yy_suc, xx_pre, yy_pre = generate_foils(index=aerofoil_index)
        plt.figure()

        plt.plot(xx_suc, yy_suc, 'r', xx_pre, yy_pre, 'b', linewidth=2)

        plt.grid(True)
        plt.xlim([0, 1])
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.gca().axis('equal')

        with open(os.path.join(input_directory, pp), 'r') as f:
            lines = f.readlines()

        parnames = lines[0].strip().split(';')
        parvals = lines[aerofoil_index + 1].strip().split(';')

        parnv = [f"{n}={float(v):.3f}" for n, v in zip(parnames, parvals)]
        parnv = [', '.join(parnv[:5]), ', '.join(parnv[5:])]

        plt.title("PARSEC airfoil with parameters:\n{}\n{}".format(parnv[0], parnv[1]))
        plt.tight_layout()
        plt.savefig(os.path.join(input_directory, f'parsec_airfoil_{type_name}_{n_samples}_{aerofoil_index}.png'))


def plot_airfoil(index, coords=np.array([]), valid=True, validity_details=None, attempt_number=None, rejection_reasons=None):
    """
    Plot an airfoil outline and its thickness distribution, then save a single figure.
    The figure is annotated with the validity result, rejection reason, and attempt count.
    """
    n_samples = params['experiment']['n_samples']
    type_name = params['experiment']['sampling_method']

    if coords.size == 0:
        input_directory = get_path('aerofoils')
        dat_path = os.path.join(input_directory, f'parsec_airfoil_{index}.dat')
        x, y = [], []
        with open(dat_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        xi, yi = float(parts[0]), float(parts[1])
                        x.append(xi)
                        y.append(yi)
                    except Exception:
                        continue
        coords = np.array(list(zip(x, y))) if x else np.empty((0, 2))

    x = coords[:, 0]
    y = coords[:, 1]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    axes[0].plot(x, y, color='tab:blue', linewidth=2)
    label = f'Sample {index}'
    if attempt_number is not None:
        label += f', attempt {attempt_number}'
    label += f' ({"VALID" if valid else "INVALID"})'
    axes[0].set_title(label)
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].grid(True, alpha=0.3)

    if validity_details is None:
        validity_details = validate_aerofoil(coords, return_details=True)[1]

    x_common = validity_details.get('x_common', np.linspace(0.0, 1.0, 500))
    thickness = validity_details.get('thickness', np.zeros_like(x_common))
    axes[1].plot(x_common, thickness, color='tab:orange', linewidth=2, label='Thickness')
    axes[1].axhline(0.04, color='tab:red', linestyle='--', linewidth=1, alpha=0.8)
    axes[1].axhline(0.25, color='tab:red', linestyle='--', linewidth=1, alpha=0.8)
    axes[1].axvline(0.10, color='tab:gray', linestyle='--', linewidth=1, alpha=0.8)
    axes[1].axvline(0.60, color='tab:gray', linestyle='--', linewidth=1, alpha=0.8)
    axes[1].set_xlabel('x/c')
    axes[1].set_ylabel('t/c')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Thickness profile')

    if rejection_reasons is None:
        rejection_reasons = _validation_reasons(validity_details)
    reason_text = '\n'.join(rejection_reasons)
    fig.text(0.02, 0.98, f"Reason: {reason_text}", va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    status_text = 'VALID' if valid else 'INVALID'
    fig.suptitle(f'Geometry check: {status_text}', y=0.98)

    save_dir = get_path('aerofoils')
    os.makedirs(save_dir, exist_ok=True)
    attempt_suffix = f'_attempt_{attempt_number}' if attempt_number is not None else ''
    save_path = os.path.join(save_dir, f'parsec_airfoil_{type_name}_{n_samples}_{index}{attempt_suffix}_valid_{valid}.png')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Airfoil plot saved to: {save_path}")


def plot_all_generated_aerofoils(test=False, max_count=None):
    """Plot every saved aerofoil with its actual validity status from the current validation rule."""
    test_type = 'test' if test else 'experiment'
    n_samples = params[test_type]['n_samples']
    aerofoil_dir = get_path('aerofoils', test=test)
    os.makedirs(aerofoil_dir, exist_ok=True)
    count = min(n_samples, max_count) if max_count is not None else n_samples

    for index in range(count):
        dat_path = os.path.join(aerofoil_dir, f'parsec_airfoil_{index}.dat')
        if not os.path.exists(dat_path):
            print(f"Skipping plot for missing file: {dat_path}")
            continue

        x, y = [], []
        with open(dat_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        xi, yi = float(parts[0]), float(parts[1])
                        x.append(xi)
                        y.append(yi)
                    except Exception:
                        continue

        if len(x) < 3:
            print(f"Skipping file with insufficient geometry data: {dat_path}")
            continue

        coords = np.array(list(zip(x, y)))
        valid, details = validate_aerofoil(coords, return_details=True)
        plot_airfoil(index, coords=coords, valid=valid, validity_details=details)

    return count


def ppointsplain(index, *args, **kwargs):
    '''Alias for ppoints that returns a string with plain data format.'''
    coords = ppoints(*args, **kwargs)
    valid, details = validate_aerofoil(coords, return_details=True)
    plot_airfoil(index, coords=coords, valid=valid, validity_details=details)
    if valid:
        coordstrlist = ["{:.6f} {:.6f}".format(coord[0], coord[1]) for coord in coords]
        return '\n'.join(coordstrlist)
    return ''


def validate_aerofoil(coord, return_details=False):
    if coord is None or len(coord) < 3:
        details = {'valid': False, 'thickness': np.array([]), 'x_common': np.array([]), 't_max': 0.0, 'x_t_max': 0.0}
        return (False, details) if return_details else False

    coord = np.asarray(coord, dtype=float)
    x = coord[:, 0]
    y = coord[:, 1]

    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        details = {'valid': False, 'thickness': np.array([]), 'x_common': np.array([]), 't_max': 0.0, 'x_t_max': 0.0}
        return (False, details) if return_details else False

    le_idx = np.argmin(np.abs(x))
    lower = coord[:le_idx + 1]
    upper = coord[le_idx:]
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]

    x_common = np.linspace(0.0, 1.0, 500)
    upper_interp = interp1d(upper[:, 0], upper[:, 1], bounds_error=False, fill_value=np.nan)
    lower_interp = interp1d(lower[:, 0], lower[:, 1], bounds_error=False, fill_value=np.nan)

    y_upper = upper_interp(x_common)
    y_lower = lower_interp(x_common)
    thickness = y_upper - y_lower

    thickness = np.asarray(thickness, dtype=float)
    valid = True
    t_max_idx = int(np.nanargmax(thickness)) if thickness.size else -1
    t_max = float(thickness[t_max_idx]) if thickness.size else 0.0
    x_t_max = float(x_common[t_max_idx]) if thickness.size else 0.0

    if np.any(~np.isfinite(thickness)) or np.any(thickness < -0.001):
        valid = False
    if t_max < 0.04 or t_max > 0.25:
        valid = False
    if x_t_max < 0.1 or x_t_max > 0.6:
        valid = False

    details = {
        'valid': valid,
        'thickness': thickness,
        'x_common': x_common,
        't_max': t_max,
        'x_t_max': x_t_max,
    }
    return (valid, details) if return_details else valid


def ppoints(cf_pre, cf_suc, npts=121, xte=1.0):
    '''
    Takes PARSEC coefficients, number of points, and returns list of
    [x,y] coordinates starting at trailing edge pressure side.
    Assumes trailing edge x position is 1.0 if not specified.
    Returns 121 points if 'npts' keyword argument not specified.
    '''
    xpts = (1 - np.cos(np.linspace(0, 1, int(np.ceil(npts/2))) * np.pi)) / 2
    xpts *= xte

    pwrs = (1/2, 3/2, 5/2, 7/2, 9/2, 11/2)
    xptsgrid = np.meshgrid(np.arange(len(pwrs)), xpts)[1]
    evalpts = lambda cf: np.sum(cf * xptsgrid**pwrs, axis=1)
    ycoords = np.append(evalpts(cf_pre)[::-1], evalpts(cf_suc)[1:])
    xcoords = np.append(xpts[::-1], xpts[1:])
    return np.array((xcoords, ycoords)).T


def _example(
    R_LE=0.01,
    x_PRE=0.45,
    y_PRE=0.060,
    d2ydx2_PRE=-0.2,
    th_PRE=2.0,
    x_SUC=0.35,
    y_SUC=0.025,
    d2ydx2_SUC=-0.35,
    th_SUC=-10.0,
    xte=1.0,
    yte=0.0,
    npts=241,
):

    cf_pre = pc.pcoef(
        xte,
        yte,
        R_LE,
        x_PRE,
        y_PRE,
        d2ydx2_PRE,
        th_PRE,
        'pre'
    )

    cf_suc = pc.pcoef(
        xte,
        yte,
        R_LE,
        x_SUC,
        y_SUC,
        d2ydx2_SUC,
        th_SUC,
        'suc'
    )

    coords = ppoints(
        cf_pre,
        cf_suc,
        npts,
        xte=xte
    )

    valid, details = validate_aerofoil(
        coords,
        return_details=True
    )

    x = coords[:, 0]
    y = coords[:, 1]

    x_common = details['x_common']
    thickness = details['thickness']

    # ==========================================================
    # Thickness derivatives
    # ==========================================================

    dt_dx = np.gradient(thickness, x_common)
    d2t_dx2 = np.gradient(dt_dx, x_common)

    zero_crossings = []
    for i in range(len(dt_dx) - 1):
        if dt_dx[i] == 0:
            zero_crossings.append(x_common[i])
        elif  dt_dx[i] * dt_dx[i+1] < 0:
            zero_crossings.append(x_common[i])

    i_max_thickness = np.argmax(thickness)
    max_thickness_x = x_common[i_max_thickness]
    max_thickness_y = thickness[i_max_thickness]


    # ==========================================================
    # Split upper/lower surfaces
    # ==========================================================

    ile = np.argmin(coords[:, 0])

    upper = coords[:ile + 1]
    lower = coords[ile:]

    # sort LE -> TE
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]

    x_grad = np.linspace(0.001, 1.0, 500)

    y_upper = np.interp(
        x_grad,
        upper[:, 0],
        upper[:, 1]
    )

    y_lower = np.interp(
        x_grad,
        lower[:, 0],
        lower[:, 1]
    )

    # ==========================================================
    # Surface gradients
    # ==========================================================

    dyu_dx = np.gradient(
        y_upper,
        x_grad
    )

    dyl_dx = np.gradient(
        y_lower,
        x_grad
    )

    grad_limit = 2.0

    dyu_dx_plot = np.clip(
        dyu_dx,
        -grad_limit,
        grad_limit
    )

    dyl_dx_plot = np.clip(
        dyl_dx,
        -grad_limit,
        grad_limit
    )
    


    # ==========================================================
    # Plotting
    # ==========================================================

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(20, 14),
        sharex=True,
        gridspec_kw={
            'height_ratios': [2, 1, 1, 1]
        }
    )

    # ==========================================================
    # Airfoil contour
    # ==========================================================

    axes[0].plot(
        x,
        y,
        linewidth=2
    )

    axes[0].axhline(
        0.0,
        color='black',
        linewidth=0.8,
        alpha=0.5
    )

    axes[0].set_ylabel('y/c')

    axes[0].set_title(
        f'Aerofoil surface | Invalid for crossing surfaces'
    )

    axes[0].grid(
        True,
        alpha=0.3
    )

    axes[0].set_xlim(
        0.0,
        1.0
    )
    axes[0].set_ylim(
        -0.1,
        0.2
    )

    # ==========================================================
    # Thickness distribution
    # ==========================================================
    axes[1].set_title(
        'Thickness distribution'
    )
    axes[1].plot(
        x_common,
        thickness,
        linewidth=2,
        color='orange'
    )

    axes[1].axhline(
        0.25,
        color='black',
        linestyle='--',
        linewidth=0.8,
        alpha=1.0
    )
    axes[1].axhline(
        0.04,
        color='black',
        linestyle='--',
        linewidth=0.8,
        alpha=1.0
    )

    axes[1].axvline(
        0.1,
        color='red',
        linestyle='--',
        linewidth=0.8,
        alpha=0.5
    )

    axes[1].axvline(
        0.6,
        color='red',
        linestyle='--',
        linewidth=0.8,
        alpha=0.5
    )

    axes[1].set_ylabel('t/c')
    axes[1].set_ylim(
        -0.1,
        0.3
    )
    axes[1].plot(max_thickness_x, max_thickness_y, 'x', color='orange', markersize=8, markeredgewidth=3)
    axes[1].annotate(f'{max_thickness_x:.2f}c', (max_thickness_x, max_thickness_y), xytext=(5,8), textcoords='offset points')

    # ==========================================================
    # Thickness gradient
    # ==========================================================
    axes[2].set_title(
        'Gradient of thickness'
    )
    axes[2].plot(
        x_common,
        dt_dx,
        linewidth=2,
        color='green'
    )

    axes[2].axhline(
        0.0,
        color='black',
        linestyle='--',
        linewidth=0.8
    )

    axes[2].set_ylabel('dt/dx')
    axes[2].set_ylim(
        -grad_limit,
        grad_limit
    )
    for xc in zero_crossings:
        axes[2].plot(xc, 0, 'gx', markersize=8, markeredgewidth=3)
        axes[2].annotate(f'{xc:.2f}c', (xc, 0), xytext=(5,8), textcoords='offset points')


    # ==========================================================
    # Surface gradients
    # ==========================================================
    axes[3].set_title(
        'Upper and lower surface gradients'
    )
    axes[3].plot(
        x_grad,
        dyu_dx_plot,
        label='Upper surface',
        linewidth=2,
        color='purple'
    )

    axes[3].plot(
        x_grad,
        dyl_dx_plot,
        label='Lower surface',
        linewidth=2,
        color='black'
    )

    axes[3].axhline(
        -0.5,
        color='black',
        linestyle='--',
        linewidth=0.8,
        alpha=1.0
    )
    axes[3].axhline(
        0.5,
        color='black',
        linestyle='--',
        linewidth=0.8,
        alpha=1.0
    )

    axes[3].axvline(
        0.2,
        color='red',
        linestyle='--',
        linewidth=0.8,
        alpha=0.5
    )

    axes[3].set_ylim(
        -grad_limit,
        grad_limit
    )

    axes[3].set_ylabel('dy/dx')
    axes[3].set_xlabel('x/c')

    axes[3].legend()

    # ==========================================================
    # Layout
    # ==========================================================

    fig.tight_layout(rect=[0.01,0.02,0.99,0.99])

    plt.show()

    return coords, details


if __name__ == "__main__":
    interrogation_aerofoil = 9
    input_directory = get_path('pre', test=False)
    test_type = 'experiment'
    n_samples = params[test_type]['n_samples']
    type_name = params[test_type]['sampling_method']
    pp = f'parsec_init_{type_name}_{n_samples}.csv'
    pparray = np.genfromtxt(
        os.path.join(input_directory, pp),
        delimiter=';',
        dtype=float,
        skip_header=1,
        ndmin=2,
    )
    aerofoil_params = {}
    for parameter, value in zip(param_bounds.keys(), pparray[interrogation_aerofoil]):
        if parameter == 'd2y/dx2_PRE':
            parameter = 'd2ydx2_PRE'
        if parameter == 'd2y/dx2_SUC':
            parameter = 'd2ydx2_SUC'
        aerofoil_params[parameter] = float(value)
    # _example(**aerofoil_params)
    _example()
