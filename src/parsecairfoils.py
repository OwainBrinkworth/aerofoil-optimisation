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
    if not isinstance(details, dict) or details.get('x_common', np.array([])).size == 0:
        return ['invalid or degenerate geometry']

    reasons = []
    if not details.get('no_intersection', False):
        reasons.append('pressure and suction surfaces intersect (thickness < 0 somewhere)')
    if not details.get('thickness_in_range', False):
        reasons.append(f"max thickness {details.get('t_max', 0.0):.4f} outside [0.04, 0.25]")
    if not details.get('single_maximum', False):
        n_turning_points = details.get('n_thickness_turning_points', 0)
        reasons.append(f"thickness distribution has {n_turning_points} turning point(s), expected exactly 1")
    if not details.get('gradients_ok', False):
        max_gradient = details.get('max_gradient_aft', 0.0)
        reasons.append(f"surface gradient {max_gradient:.3f} exceeds 0.5 aft of x/c=0.2")

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


def validate_aerofoil(coord, return_details=False):
    """
    Check whether an aerofoil shape is physically sensible.

    Four criteria are checked:
        1. No intersection of the pressure/suction surfaces: t(x) >= 0
           for all x in [0, 1], where t(x) is the thickness distribution.
        2. Maximum thickness is between 4% and 25% of the chord:
           0.04 <= max(t(x)) <= 0.25.
        3. The thickness distribution has a single maximum: dt/dx = 0 has
           exactly one solution for x in (0, 1).
        4. The upper/lower surface gradients are not too steep aft of
           x/c = 0.2: max(|dy/dx|) <= 0.5 for x in [0.2, 1].
    """
    empty_details = {
        'valid': False, 'thickness': np.array([]), 'x_common': np.array([]),
        't_max': 0.0, 'x_t_max': 0.0, 'dt_dx': np.array([]),
        'n_thickness_turning_points': 0, 'thickness_turning_points': [],
        'x_grad': np.array([]), 'dyu_dx': np.array([]), 'dyl_dx': np.array([]),
        'max_gradient_aft': 0.0,
        'no_intersection': False, 'thickness_in_range': False,
        'single_maximum': False, 'gradients_ok': False,
    }

    if coord is None or len(coord) < 3:
        return (False, empty_details) if return_details else False

    coord = np.asarray(coord, dtype=float)
    x, y = coord[:, 0], coord[:, 1]
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        return (False, empty_details) if return_details else False

    # Split the surface into upper (suction) and lower (pressure) halves
    # at the leading edge, each sorted from leading to trailing edge.
    le_idx = np.argmin(np.abs(x))
    lower = coord[:le_idx + 1]
    upper = coord[le_idx:]
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]

    # --- Thickness distribution, t(x) = y_upper(x) - y_lower(x) ---
    x_common = np.linspace(0.0, 1.0, 500)
    y_upper = interp1d(upper[:, 0], upper[:, 1], bounds_error=False, fill_value=np.nan)(x_common)
    y_lower = interp1d(lower[:, 0], lower[:, 1], bounds_error=False, fill_value=np.nan)(x_common)
    thickness = y_upper - y_lower
    finite_thickness = bool(np.isfinite(thickness).all())

    t_max_idx = int(np.nanargmax(thickness)) if finite_thickness else -1
    t_max = float(thickness[t_max_idx]) if finite_thickness else 0.0
    x_t_max = float(x_common[t_max_idx]) if finite_thickness else 0.0

    # Criterion 1: no intersection -> t(x) >= 0 everywhere (a tiny negative
    # tolerance is allowed to absorb interpolation/floating-point noise).
    no_intersection = finite_thickness and bool(np.all(thickness >= -1e-6))

    # Criterion 2: max thickness between 4% and 25% chord.
    thickness_in_range = 0.04 <= t_max <= 0.25

    # Criterion 3: the thickness distribution has a single maximum, i.e.
    # dt/dx crosses zero exactly once strictly inside (0, 1).
    dt_dx = np.gradient(thickness, x_common) if finite_thickness else np.array([])
    interior = (x_common > 0.0) & (x_common < 1.0)
    thickness_turning_points = []
    dt_interior = dt_dx[interior]
    x_interior = x_common[interior]
    for i in range(len(dt_interior) - 1):
        if dt_interior[i] == 0 or dt_interior[i] * dt_interior[i + 1] < 0:
            thickness_turning_points.append(float(x_interior[i]))
    single_maximum = finite_thickness and len(thickness_turning_points) == 1

    # --- Surface gradients dy/dx, used for criterion 4 ---
    x_grad = np.linspace(0.0, 1.0, 500)
    y_upper_grad = np.interp(x_grad, upper[:, 0], upper[:, 1])
    y_lower_grad = np.interp(x_grad, lower[:, 0], lower[:, 1])
    dyu_dx = np.gradient(y_upper_grad, x_grad)
    dyl_dx = np.gradient(y_lower_grad, x_grad)

    # Criterion 4: gradients must not be too steep aft of x/c = 0.2.
    aft = x_grad >= 0.2
    max_gradient_aft = float(np.max(np.abs(np.concatenate([dyu_dx[aft], dyl_dx[aft]]))))
    gradients_ok = max_gradient_aft <= 0.5

    valid = no_intersection and thickness_in_range and single_maximum and gradients_ok

    details = {
        'valid': valid,
        'thickness': thickness,
        'x_common': x_common,
        't_max': t_max,
        'x_t_max': x_t_max,
        'dt_dx': dt_dx,
        'n_thickness_turning_points': len(thickness_turning_points),
        'thickness_turning_points': thickness_turning_points,
        'x_grad': x_grad,
        'dyu_dx': dyu_dx,
        'dyl_dx': dyl_dx,
        'max_gradient_aft': max_gradient_aft,
        'no_intersection': no_intersection,
        'thickness_in_range': thickness_in_range,
        'single_maximum': single_maximum,
        'gradients_ok': gradients_ok,
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



def parsec_params_to_coords(parsec_params, npts=241, xte=1.0, yte=0.0):
    """
    Convert a dict of named PARSEC parameters into (x, y) surface coordinates.

    Parameters:
        parsec_params (dict): must contain R_LE, x_PRE, y_PRE, d2ydx2_PRE,
            th_PRE, x_SUC, y_SUC, d2ydx2_SUC, th_SUC.
        npts (int): number of points on the aerofoil surface.
        xte, yte (float): trailing-edge x/y position (normally 1.0 and 0.0).

    Returns:
        np.ndarray: Nx2 array of (x, y) coordinates, trailing edge to
            trailing edge via the lower (pressure) then upper (suction)
            surface.
    """
    cf_pre = pc.pcoef(
        xte, yte,
        parsec_params['R_LE'],
        parsec_params['x_PRE'],
        parsec_params['y_PRE'],
        parsec_params['d2ydx2_PRE'],
        parsec_params['th_PRE'],
        'pre',
    )
    cf_suc = pc.pcoef(
        xte, yte,
        parsec_params['R_LE'],
        parsec_params['x_SUC'],
        parsec_params['y_SUC'],
        parsec_params['d2ydx2_SUC'],
        parsec_params['th_SUC'],
        'suc',
    )
    return ppoints(cf_pre, cf_suc, npts, xte=xte)


def plot_aerofoil_diagnostics(parsec_params, npts=241, show=False, save_path=None):
    """
    Generate an aerofoil from a set of PARSEC parameters and plot a set of
    diagnostic figures, one per validity criterion, so it's easy to see
    exactly why an aerofoil passed or failed:

        1. The aerofoil contour (upper + lower surface).
        2. The thickness distribution (t/c vs x/c) — criteria 1 & 2:
           t(x) >= 0 everywhere, and 0.04 <= max(t) <= 0.25.
        3. The gradient of the thickness distribution — criterion 3: dt/dx
           should cross zero exactly once (a single maximum), marked here.
        4. The upper/lower surface gradients (dy/dx) — criterion 4:
           |dy/dx| <= 0.5 aft of x/c = 0.2.

    Parameters:
        parsec_params (dict): PARSEC parameters, see parsec_params_to_coords.
        npts (int): number of points used to define the aerofoil surface.
        show (bool): if True, display the figure interactively (plt.show()).
            Defaults to False so this can be called from a script without
            blocking.
        save_path (str or None): if given, save the figure to this path
            (300 dpi, suitable for a report).

    Returns:
        (coords, details): the aerofoil coordinates and the validity-check
            details dict returned by validate_aerofoil.
    """
    coords = parsec_params_to_coords(parsec_params, npts=npts)
    valid, details = validate_aerofoil(coords, return_details=True)

    x, y = coords[:, 0], coords[:, 1]
    x_common = details['x_common']
    thickness = details['thickness']
    dt_dx = details['dt_dx']
    turning_points = details['thickness_turning_points']
    x_grad = details['x_grad']

    # Clip the displayed gradients to a sensible y-axis range; the actual
    # pass/fail threshold (0.5) is drawn as a horizontal line regardless.
    grad_limit = 2.0
    dyu_dx_plot = np.clip(details['dyu_dx'], -grad_limit, grad_limit)
    dyl_dx_plot = np.clip(details['dyl_dx'], -grad_limit, grad_limit)

    t_max_idx = np.argmax(thickness)

    # --- Plotting ---
    fig, axes = plt.subplots(
        4, 1, figsize=(10, 12), sharex=True,
        gridspec_kw={'height_ratios': [2, 1, 1, 1]},
    )

    # 1. Aerofoil contour
    status = 'VALID' if valid else 'INVALID'
    axes[0].plot(x, y, linewidth=2, color='tab:blue')
    axes[0].axhline(0.0, color='black', linewidth=0.8, alpha=0.5)
    axes[0].set_title(f'Aerofoil contour ({status})')
    axes[0].set_ylabel('y/c [-]')
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(-0.1, 0.2)
    axes[0].grid(True, alpha=0.3)

    # 2. Thickness distribution — criteria 1 (t(x) >= 0) & 2 (0.04-0.25 max)
    axes[1].plot(x_common, thickness, linewidth=2, color='tab:orange')
    axes[1].axhline(0.0, color='tab:red', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[1].axhline(0.25, color='black', linestyle='--', linewidth=0.8)
    axes[1].axhline(0.04, color='black', linestyle='--', linewidth=0.8)
    axes[1].plot(x_common[t_max_idx], thickness[t_max_idx], 'x', color='tab:orange', markersize=8, markeredgewidth=3)
    axes[1].annotate(f'{x_common[t_max_idx]:.2f}c', (x_common[t_max_idx], thickness[t_max_idx]), xytext=(5, 8), textcoords='offset points')
    axes[1].set_title(f"Thickness distribution (max={details['t_max']:.3f} at x/c={details['x_t_max']:.2f})")
    axes[1].set_ylabel('t/c [-]')
    axes[1].set_ylim(-0.1, 0.3)
    axes[1].grid(True, alpha=0.3)

    # 3. Gradient of thickness — criterion 3 (exactly one turning point)
    axes[2].plot(x_common, dt_dx, linewidth=2, color='tab:green')
    axes[2].axhline(0.0, color='black', linestyle='--', linewidth=0.8)
    for xc in turning_points:
        axes[2].plot(xc, 0, 'x', color='tab:green', markersize=8, markeredgewidth=3)
        axes[2].annotate(f'{xc:.2f}c', (xc, 0), xytext=(5, 8), textcoords='offset points')
    axes[2].set_title(f'Gradient of thickness ({len(turning_points)} turning point(s), expect 1)')
    axes[2].set_ylabel('d(t/c)/d(x/c)')
    axes[2].set_ylim(-grad_limit, grad_limit)
    axes[2].grid(True, alpha=0.3)

    # 4. Upper/lower surface gradients — criterion 4 (|dy/dx| <= 0.5 for x >= 0.2)
    axes[3].plot(x_grad, dyu_dx_plot, linewidth=2, color='tab:purple', label='Upper surface')
    axes[3].plot(x_grad, dyl_dx_plot, linewidth=2, color='black', label='Lower surface')
    axes[3].axhline(-0.5, color='black', linestyle='--', linewidth=0.8)
    axes[3].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
    axes[3].axvline(0.2, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[3].set_title(f"Surface gradients (max |dy/dx| aft of x/c=0.2 is {details['max_gradient_aft']:.3f})")
    axes[3].set_ylabel('dy/dx')
    axes[3].set_xlabel('x/c [-]')
    axes[3].set_ylim(-grad_limit, grad_limit)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Aerofoil diagnostics plot saved to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return coords, details
