"""
XFOIL validation library.

This module provides the building blocks used by validate_xfoil.py to check
that our XFOIL setup (NACA 0012, Re=2e6, Mach=0.15) matches published wind
tunnel data, and to understand how XFOIL's solver settings affect the
result. It is a library only — run validate_xfoil.py to actually produce
results and plots.

Two validation studies are supported:

    1. Ncrit sensitivity: run a full CL vs angle-of-attack polar sweep for
       several Ncrit (transition amplification factor) values, and compare
       each one against NASA/Ladson experimental data (RMSE, lift curve
       plot).
    2. Iteration sensitivity: run XFOIL at a handful of individual angles of
       attack for several iteration limits, and plot how the residual (rms)
       converges with iteration number. This is used to choose a sensible
       maximum iteration count — no experimental comparison is needed here.
"""

import os
import re as regex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import get_path


# ============================================================
# NASA / Ladson NACA 0012 experimental data
# NASA TM-4074, Charles L. Ladson, 1988
# Free-transition data, M = 0.15
# ============================================================

EXPERIMENTAL_DATA = {
    2.00e6: {
        "alpha": np.array([
            -4.25, -2.10, 0.00, 1.85, 4.25, 6.05, 8.15,
            10.15, 11.15, 12.10, 13.08, 14.25, 15.25, 16.25,
        ]),
        "cl": np.array([
            -0.4300, -0.2150, 0.0000, 0.1940, 0.4450, 0.6250,
            0.8550, 1.0450, 1.1350, 1.2180, 1.2900, 1.3620,
            1.4080, 0.7530,
        ]),
    },
}


# ============================================================
# Default solver configuration
# ============================================================

REYNOLDS_NUMBER = 2.00e6
MACH_NUMBER = 0.15

# Full polar sweep used for the Ncrit study.
AOA = np.arange(-5.0, 20.0, 0.5)

# A handful of individual angles used for the iteration/residual study.
# Chosen to sample attached flow, pre-stall and post-stall behaviour.
RESIDUAL_AOA = np.array([-4.0, -2.0, 0.0, 4.0, 8.0, 10.0, 12.0, 14.0, 15.0, 16.0, 17.0, 18.0])

# Baseline settings: held constant while the *other* parameter is varied.
BASE_ITERATIONS = 100
BASE_NCRIT = 11

ITERATION_VALUES = [25, 50, 100, 200]
NCRIT_VALUES = [5, 7, 9, 11]

PRINT_XFOIL_OUTPUT = False
XFOIL_TIMEOUT_SECONDS = 300

XFOIL_EXE = os.path.join(get_path("xfoil"), "bin", "xfoil")


# ============================================================
# Report-quality plot styling
# ============================================================

def _style_axes(ax, xlabel, ylabel, title=None):
    """Apply a consistent, report-friendly style to a plot axis."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)


def _save_figure(fig, save_path, show=False):
    """Save a figure at report quality (300 dpi) and optionally display it."""
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


# ============================================================
# XFOIL input construction
# ============================================================

def build_xfoil_commands(polar_file, aoa, re, mach, iterations, ncrit):
    """Build an XFOIL input deck for a NACA 0012 polar sweep."""
    commands = "NACA 0012\n"
    commands += "PANE\n"
    commands += "OPER\n"
    commands += f"VISC {int(re)}\n"
    commands += f"MACH {mach}\n"
    commands += f"ITER {int(iterations)}\n"

    # Set the transition (amplification factor) criterion.
    commands += "VPAR\n"
    commands += f"N {float(ncrit)}\n"
    commands += "\n"

    # Start polar accumulation, run every requested angle, then stop.
    commands += "PACC\n"
    commands += f"{polar_file}\n"
    commands += "\n"
    for alpha in aoa:
        commands += f"ALFA {alpha:.4f}\n"
    commands += "\n"
    commands += "PACC\n"
    commands += "QUIT\n"
    return commands


# ============================================================
# Parse XFOIL console output
# ============================================================

def parse_residual_history(stdout):
    """
    Extract (iteration, rms residual) pairs from XFOIL console output.

    Expected line format resembles: "  1   rms: 0.1234 ...". XFOIL versions
    can format this differently — an empty result means nothing matched
    this parser, not necessarily that the run failed.
    """
    pattern = regex.compile(
        r"^\s*(\d+)\s+rms:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
        flags=regex.IGNORECASE,
    )

    rows = []
    for line in stdout.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        iteration = int(match.group(1))
        rms = float(match.group(2))
        if np.isfinite(rms) and rms > 0:
            rows.append({"iteration": iteration, "rms": rms})

    return pd.DataFrame(rows, columns=["iteration", "rms"])


def has_convergence_failure(stdout):
    """Detect XFOIL's explicit viscous convergence failure message."""
    return bool(regex.search(r"VISCAL:\s*Convergence failed", stdout, flags=regex.IGNORECASE))


# ============================================================
# Run XFOIL: full polar sweep (used by the Ncrit study)
# ============================================================

def _run_xfoil_process(commands, output_dir, case_name):
    """Run XFOIL with the given input deck and return (stdout, stderr, returncode)."""
    try:
        process = subprocess.run(
            [XFOIL_EXE], input=commands, text=True, capture_output=True,
            cwd=output_dir, timeout=XFOIL_TIMEOUT_SECONDS,
        )
        return process.stdout or "", process.stderr or "", process.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        raise RuntimeError(f"XFOIL timed out for {case_name}") from exc


def run_xfoil(case_name, aoa, re, mach, iterations, ncrit, output_dir):
    """Run a full AoA sweep in one XFOIL process and save its polar/log files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    polar_name = f"{case_name}.polar"
    polar_path = output_dir / polar_name
    input_path = output_dir / f"{case_name}.inp"
    log_path = output_dir / f"{case_name}.log"

    # Prevent an old polar being mistaken for a new result.
    if polar_path.exists():
        polar_path.unlink()

    commands = build_xfoil_commands(polar_name, aoa, re, mach, iterations, ncrit)
    input_path.write_text(commands, newline="\r\n")

    print(f"\nRunning {case_name}: Re={re:.2e}, Mach={mach}, ITER={iterations}, Ncrit={ncrit}")
    stdout, stderr, returncode = _run_xfoil_process(commands, output_dir, case_name)
    log_path.write_text(stdout + "\n\n--- STDERR ---\n" + stderr)

    if PRINT_XFOIL_OUTPUT:
        print(stdout)
    if stderr.strip():
        print(f"XFOIL stderr for {case_name}:\n{stderr}")
    if returncode != 0:
        raise RuntimeError(f"XFOIL exited with code {returncode} for {case_name}. See {log_path}")
    if not polar_path.is_file():
        raise FileNotFoundError(f"XFOIL did not create {polar_path}. See {log_path}")

    residual_history = parse_residual_history(stdout)
    if residual_history.empty:
        print(f"WARNING: no residual history parsed for {case_name}. Inspect {log_path}")

    return {
        "case": case_name,
        "polar_path": str(polar_path),
        "log_path": str(log_path),
        "stdout": stdout,
        "return_code": returncode,
        "convergence_warning": has_convergence_failure(stdout),
        "residual_history": residual_history,
    }


# ============================================================
# Run XFOIL: single angle of attack (used by the iteration study)
# ============================================================

def run_xfoil_single_alpha_residual(alpha, case_name, iterations, ncrit, output_dir):
    """
    Run XFOIL at one angle of attack in its own process, so the console log
    contains exactly one residual history (easier to read/plot).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_tag = f"a{alpha:+06.2f}".replace(".", "p")
    polar_name = f"{case_name}_{alpha_tag}.polar"
    log_path = output_dir / f"{case_name}_{alpha_tag}.log"
    polar_path = output_dir / polar_name

    if polar_path.exists():
        polar_path.unlink()

    commands = build_xfoil_commands(polar_name, [alpha], REYNOLDS_NUMBER, MACH_NUMBER, iterations, ncrit)
    (output_dir / f"{case_name}_{alpha_tag}.inp").write_text(commands, newline="\r\n")

    try:
        stdout, stderr, returncode = _run_xfoil_process(commands, output_dir, case_name)
    except RuntimeError:
        log_path.write_text("XFOIL timed out")
        return {"alpha": alpha, "history": pd.DataFrame(columns=["iteration", "rms"]), "failed": True, "reason": "timeout", "log_path": str(log_path)}

    log_path.write_text(stdout + "\n\n--- STDERR ---\n" + stderr)

    history = parse_residual_history(stdout)
    failed = has_convergence_failure(stdout) or returncode != 0
    if history.empty:
        reason = "no residuals parsed"
    elif failed:
        reason = "explicit solver failure or non-zero return code"
    else:
        reason = "no explicit failure detected"

    return {"alpha": alpha, "history": history, "failed": failed, "reason": reason, "log_path": str(log_path)}


# ============================================================
# Read and validate XFOIL polar output
# ============================================================

def read_xfoil_polar(filename):
    """Read alpha, CL and CD columns from an XFOIL polar file."""
    lines = Path(filename).read_text(errors="replace").splitlines()

    header_line = next(
        (i for i, line in enumerate(lines) if "alpha" in line.lower() and "CL" in line),
        None,
    )
    if header_line is None:
        raise ValueError(f"Could not find XFOIL polar header in {filename}")

    data = []
    for line in lines[header_line + 1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            alpha, cl, cd = map(float, parts[:3])
        except ValueError:
            continue
        data.append([alpha, cl, cd])

    if not data:
        raise ValueError(f"No numerical polar data found in {filename}")

    df = pd.DataFrame(data, columns=["alpha", "cl", "cd"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["alpha", "cl", "cd"])
    return df.sort_values("alpha").drop_duplicates("alpha").reset_index(drop=True)


def diagnose_polar(xfoil_data, requested_aoa, stdout):
    """Compare the angles XFOIL actually returned against what was requested."""
    requested_rounded = np.round(np.asarray(requested_aoa, dtype=float), 3)
    returned = np.round(xfoil_data["alpha"].to_numpy(dtype=float), 3)

    missing = [float(a) for a in requested_rounded if not np.any(np.isclose(returned, a, atol=1e-3))]

    cl = xfoil_data["cl"].to_numpy()
    cd = xfoil_data["cd"].to_numpy()
    invalid_rows = ~np.isfinite(cl) | ~np.isfinite(cd) | (cd < 0)

    return {
        "missing_alpha": missing,
        "missing_count": len(missing),
        "invalid_row_count": int(np.sum(invalid_rows)),
        "solver_failure_message": has_convergence_failure(stdout),
        "all_requested_points_present": len(missing) == 0,
    }


def polar_with_gaps(xfoil_data, requested_aoa):
    """
    Map the polar onto the requested AoA grid, leaving NaN for any angle
    XFOIL didn't return. Matplotlib will then break the line at a NaN
    instead of joining straight across a missing/failed angle.
    """
    grid = pd.DataFrame({"alpha": np.asarray(requested_aoa, dtype=float)})
    data = xfoil_data.copy()
    grid["key"] = grid["alpha"].round(3)
    data["key"] = data["alpha"].round(3)
    merged = grid.merge(data[["key", "cl"]], on="key", how="left")
    return merged["alpha"].to_numpy(), merged["cl"].to_numpy()


def calculate_errors(xfoil_data, experimental_data, requested_aoa):
    """
    Calculate CL error metrics at the experimental AoA points only.

    Interpolates between adjacent XFOIL grid points when an experimental
    angle doesn't land exactly on the requested grid, but never
    extrapolates past a missing/failed point.
    """
    alpha_grid, cl_grid = polar_with_gaps(xfoil_data, requested_aoa)
    alpha_exp = np.asarray(experimental_data["alpha"], dtype=float)
    cl_exp = np.asarray(experimental_data["cl"], dtype=float)

    alpha_valid, cl_exp_valid, cl_calc_valid = [], [], []
    for a_exp, cl_e in zip(alpha_exp, cl_exp):
        if a_exp < alpha_grid[0] or a_exp > alpha_grid[-1]:
            continue

        right = int(np.searchsorted(alpha_grid, a_exp, side="left"))

        if right < len(alpha_grid) and np.isclose(alpha_grid[right], a_exp, atol=1e-8):
            if not np.isfinite(cl_grid[right]):
                continue
            cl_calc = cl_grid[right]
        else:
            left = right - 1
            if left < 0 or right >= len(alpha_grid):
                continue
            if not (np.isfinite(cl_grid[left]) and np.isfinite(cl_grid[right])):
                continue
            if alpha_grid[right] - alpha_grid[left] > 0.51:
                continue
            cl_calc = np.interp(a_exp, [alpha_grid[left], alpha_grid[right]], [cl_grid[left], cl_grid[right]])

        alpha_valid.append(a_exp)
        cl_exp_valid.append(cl_e)
        cl_calc_valid.append(cl_calc)

    if not alpha_valid:
        return {"rmse": np.nan, "mae": np.nan, "max_error": np.nan, "n_points": 0}

    error = np.asarray(cl_calc_valid) - np.asarray(cl_exp_valid)
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "max_error": float(np.max(np.abs(error))),
        "n_points": len(error),
    }


# ============================================================
# Plot: CL vs AoA, for the Ncrit study
# ============================================================

def plot_lift_curves(results, experimental_data, requested_aoa, title, plot_path, show=False):
    """Plot CL vs AoA for each case, overlaid with experimental data."""
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        experimental_data["alpha"], experimental_data["cl"], "ko",
        markersize=5, label="NASA Langley / Ladson experiment", zorder=5,
    )

    for result in results:
        alpha_plot, cl_plot = polar_with_gaps(result["xfoil"], requested_aoa)
        rmse = result["error"]["rmse"]
        label = (
            f"{result['case']} (RMSE={rmse:.4f}, N={result['error']['n_points']})"
            if np.isfinite(rmse) else f"{result['case']} (RMSE=N/A)"
        )
        ax.plot(alpha_plot, cl_plot, "-", linewidth=1.5, label=label)

    _style_axes(ax, r"Angle of attack, $\alpha$ [deg]", r"Lift coefficient, $C_L$ [-]", title)
    ax.legend(fontsize=8)
    _save_figure(fig, plot_path, show=show)


# ============================================================
# Plot: residual convergence vs iteration, for the iteration study
# ============================================================

def plot_iteration_residuals(cases, output_dir, aoa_values=RESIDUAL_AOA, show=False):
    """
    For each iteration-limit case, run every angle in aoa_values and plot
    its residual history (RMS residual vs iteration number). Each curve is
    labelled with how many iterations it took to converge, which is what
    you need to pick a sensible ITER setting.
    """
    output_dir = Path(output_dir)
    residual_dir = output_dir / "single_alpha_logs"
    residual_dir.mkdir(parents=True, exist_ok=True)

    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, 1, figsize=(10, max(5, 3.2 * n_cases)), squeeze=False)
    axes = axes.ravel()

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(np.min(aoa_values)), vmax=float(np.max(aoa_values)))
    diagnostic_rows = []

    for ax, case in zip(axes, cases):
        print(f"\nResidual diagnostics: {case['case']}")
        for alpha in aoa_values:
            run = run_xfoil_single_alpha_residual(
                alpha=float(alpha), case_name=case["case"],
                iterations=case["iterations"], ncrit=case["ncrit"],
                output_dir=residual_dir,
            )
            history = run["history"]
            last_iteration = int(history["iteration"].max()) if not history.empty else np.nan

            diagnostic_rows.append({
                "case": case["case"], "iterations_limit": case["iterations"], "ncrit": case["ncrit"],
                "alpha": alpha, "residual_records": len(history),
                "last_iteration_recorded": last_iteration,
                "final_rms": float(history["rms"].iloc[-1]) if not history.empty else np.nan,
                "failed": run["failed"], "reason": run["reason"], "log_file": run["log_path"],
            })

            if history.empty:
                print(f"  AoA={alpha:5.1f}: no residual history parsed; inspect {run['log_path']}")
                continue

            ax.semilogy(
                history["iteration"], history["rms"], color=cmap(norm(alpha)),
                linewidth=1.2, alpha=0.85,
                label=f"AoA = {alpha:g}deg ({int(last_iteration)} iterations)",
            )

        _style_axes(ax, "Iteration", "RMS residual [-]", f"{case['case']} | ITER limit = {case['iterations']}, Ncrit = {case['ncrit']}")
        ax.legend(title="AoA (iterations to converge)", fontsize=7, title_fontsize=8, loc="best")

    fig.suptitle("NACA 0012 — residual convergence vs iteration count", y=1.0)

    plot_path = output_dir / "iterations_residuals_by_alpha.png"
    _save_figure(fig, plot_path, show=show)

    diagnostics = pd.DataFrame(diagnostic_rows)
    csv_path = output_dir / "iterations_residual_diagnostics.csv"
    diagnostics.to_csv(csv_path, index=False)
    print(f"Residual diagnostics saved to: {csv_path}")
    return diagnostics


# ============================================================
# Run a full study: one XFOIL polar sweep per case, with error metrics
# ============================================================

def run_study(study_name, description, cases, experimental_data, show=False):
    """
    Run a full AoA sweep for each case, compare it against experimental
    data, and produce a combined lift-curve plot and summary CSV.
    """
    output_dir = Path(get_path("results_validation", description=description))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case in cases:
        run = run_xfoil(
            case_name=case["case"], aoa=AOA, re=REYNOLDS_NUMBER, mach=MACH_NUMBER,
            iterations=case["iterations"], ncrit=case["ncrit"], output_dir=output_dir,
        )
        xfoil_data = read_xfoil_polar(run["polar_path"])
        diagnostics = diagnose_polar(xfoil_data, AOA, run["stdout"])
        error = calculate_errors(xfoil_data, experimental_data, AOA)

        results.append({**case, "run": run, "xfoil": xfoil_data, "diagnostics": diagnostics, "error": error})
        print(
            f"  RMSE={error['rmse']:.5f} (experimental points={error['n_points']}), "
            f"missing AoA={diagnostics['missing_count']}, solver warning={diagnostics['solver_failure_message']}"
        )

    summary = pd.DataFrame([
        {
            "study": study_name,
            "case": r["case"],
            "iterations": r["iterations"],
            "ncrit": r["ncrit"],
            "rmse_cl": r["error"]["rmse"],
            "mae_cl": r["error"]["mae"],
            "max_abs_error_cl": r["error"]["max_error"],
            "n_experimental_points": r["error"]["n_points"],
            "missing_aoa_count": r["diagnostics"]["missing_count"],
            "solver_failure_message": r["diagnostics"]["solver_failure_message"],
            "polar_file": r["run"]["polar_path"],
            "log_file": r["run"]["log_path"],
        }
        for r in results
    ])
    summary_path = output_dir / f"{study_name}_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_lift_curves(
        results=results, experimental_data=experimental_data, requested_aoa=AOA,
        title=f"NACA 0012 — {study_name} study: XFOIL vs Ladson experimental data (Re={REYNOLDS_NUMBER:.1e})",
        plot_path=output_dir / f"{study_name}_cl_vs_alpha.png",
        show=show,
    )

    print(f"Summary CSV saved to: {summary_path}")
    return results, summary, output_dir
