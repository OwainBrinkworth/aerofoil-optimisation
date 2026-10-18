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

# Configuration

# ============================================================



REYNOLDS_NUMBER = 2.00e6

MACH_NUMBER = 0.15



# Main polar sweeps

AOA = np.arange(-5.0, 20.0, 0.5)



# Baseline parameters: vary one parameter at a time

BASE_ITERATIONS = 100

BASE_NCRIT = 11



ITERATION_VALUES = [25, 50, 100, 200]

NCRIT_VALUES = [5, 7, 9, 11]



# Panel sensitivity.

#

# IMPORTANT:

# Check the PPAR command meanings and accepted values for your

# installed XFOIL version before using the refined cases.

#

# "default" uses XFOIL's default panel distribution.

# N = requested panel count; T and R are panel-distribution

# parameters in builds supporting these PPAR controls.

PANEL_CASES = {

    "default": None,

    "refined_240": {"n": 240, "t": 0.15, "r": 0.20},

    "refined_300": {"n": 300, "t": 0.10, "r": 0.15},

}



STUDY_DESCRIPTIONS = {

    "iterations": "05_iteration_sweep",

    "ncrit": "06_ncrit_sweep",

    "panels": "07_panel_sensitivity",

}



# Residual diagnostics use independent runs at each angle.

#

# Set to AOA to run every half-degree angle, but this can create

# hundreds of XFOIL processes. The default below samples the

# attached-flow, pre-stall and post-stall regions.

#

# Include more angles here if you want finer resolution near stall.

RESIDUAL_AOA = np.array([

    -4.0, -2.0, 0.0, 4.0, 8.0, 10.0,

    12.0, 14.0, 15.0, 16.0, 17.0, 18.0,

])



PRINT_XFOIL_OUTPUT = False

XFOIL_TIMEOUT_SECONDS = 300



XFOIL_EXE = os.path.join(

    get_path("xfoil"),

    "bin",

    "xfoil",

)





# ============================================================

# XFOIL input construction

# ============================================================



def build_xfoil_commands(

    polar_file,

    aoa,

    re,

    mach,

    iterations,

    ncrit,

    panel_settings=None,

):

    """Build an XFOIL input deck for a NACA 0012 simulation."""



    commands = "NACA 0012\n"



    # Optional panel-distribution settings.

    if panel_settings is not None:

        commands += "PPAR\n"

        commands += f"N {panel_settings['n']}\n"

        commands += f"T {panel_settings['t']}\n"

        commands += f"R {panel_settings['r']}\n"

        commands += "\n"



    commands += "PANE\n"

    commands += "OPER\n"

    commands += f"VISC {int(re)}\n"

    commands += f"MACH {mach}\n"

    commands += f"ITER {int(iterations)}\n"



    # Set transition amplification factor.

    commands += "VPAR\n"

    commands += f"N {float(ncrit)}\n"

    commands += "\n"



    # Start polar accumulation.

    commands += "PACC\n"

    commands += f"{polar_file}\n"

    commands += "\n"



    # AoA points are run in sequence.

    for alpha in aoa:

        commands += f"ALFA {alpha:.4f}\n"



    # Stop polar accumulation and exit.

    commands += "\n"

    commands += "PACC\n"

    commands += "QUIT\n"



    return commands





# ============================================================

# Parse XFOIL residual history

# ============================================================



def parse_residual_history(stdout):

    """

    Extract iteration number and RMS residual from XFOIL output.



    Expected line format resembles:

        1 rms: 0.1234 ...



    XFOIL versions can format output differently. An empty result

    means no lines matched this parser; it does not prove failure.

    """



    rows = []



    pattern = regex.compile(

        r"^\s*(\d+)\s+rms:\s*"

        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",

        flags=regex.IGNORECASE,

    )



    for line in stdout.splitlines():

        match = pattern.search(line)



        if not match:

            continue



        iteration = int(match.group(1))

        rms = float(match.group(2))



        if np.isfinite(rms) and rms > 0:

            rows.append({

                "iteration": iteration,

                "rms": rms,

            })



    return pd.DataFrame(

        rows,

        columns=["iteration", "rms"],

    )





def has_convergence_failure(stdout):

    """Detect XFOIL's explicit viscous convergence failure message."""



    return bool(

        regex.search(

            r"VISCAL:\s*Convergence failed",

            stdout,

            flags=regex.IGNORECASE,

        )

    )





# ============================================================

# Run XFOIL: sequential polar sweep

# ============================================================



def run_xfoil(

    case_name,

    aoa,

    re,

    mach,

    iterations,

    ncrit,

    output_dir,

    panel_settings=None,

):

    """Run an AoA sweep and save its polar, input and console log."""



    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)



    polar_name = f"{case_name}.polar"

    polar_path = output_dir / polar_name

    input_path = output_dir / f"{case_name}.inp"

    log_path = output_dir / f"{case_name}.log"



    # Prevent an old polar being mistaken for a new result.

    if polar_path.exists():

        polar_path.unlink()



    commands = build_xfoil_commands(

        polar_file=polar_name,

        aoa=aoa,

        re=re,

        mach=mach,

        iterations=iterations,

        ncrit=ncrit,

        panel_settings=panel_settings,

    )



    input_path.write_text(commands, newline="\r\n")



    print(f"\nRunning {case_name}")

    print(

        f"  Re={re:.2e}, Mach={mach}, ITER={iterations}, "

        f"Ncrit={ncrit}, panels={panel_settings or 'default'}"

    )



    try:

        process = subprocess.run(

            [XFOIL_EXE],

            input=commands,

            text=True,

            capture_output=True,

            cwd=output_dir,

            timeout=XFOIL_TIMEOUT_SECONDS,

        )

        stdout = process.stdout or ""

        stderr = process.stderr or ""



    except subprocess.TimeoutExpired as exc:

        stdout = exc.stdout or ""

        stderr = exc.stderr or ""



        if isinstance(stdout, bytes):

            stdout = stdout.decode(errors="replace")

        if isinstance(stderr, bytes):

            stderr = stderr.decode(errors="replace")



        log_path.write_text(

            stdout + "\n\n--- STDERR ---\n" + stderr

        )

        raise RuntimeError(

            f"XFOIL timed out for {case_name}. Log: {log_path}"

        ) from exc



    log_path.write_text(

        stdout + "\n\n--- STDERR ---\n" + stderr

    )



    if PRINT_XFOIL_OUTPUT:

        print(stdout)



    if stderr.strip():

        print(f"XFOIL stderr for {case_name}:\n{stderr}")



    if process.returncode != 0:

        raise RuntimeError(

            f"XFOIL exited with code {process.returncode} "

            f"for {case_name}. See {log_path}"

        )



    if not polar_path.is_file():

        raise FileNotFoundError(

            f"XFOIL did not create {polar_path}. See {log_path}"

        )



    residual_history = parse_residual_history(stdout)



    if residual_history.empty:

        print(

            f"WARNING: no residual history parsed for {case_name}. "

            f"Inspect {log_path}"

        )



    return {

        "case": case_name,

        "polar_path": str(polar_path),

        "input_path": str(input_path),

        "log_path": str(log_path),

        "stdout": stdout,

        "return_code": process.returncode,

        "convergence_warning": has_convergence_failure(stdout),

        "residual_history": residual_history,

        "panel_settings": panel_settings,

    }





# ============================================================

# Read XFOIL polar

# ============================================================



def read_xfoil_polar(filename):

    """Read alpha, CL and CD from an XFOIL polar file."""



    lines = Path(filename).read_text(

        errors="replace"

    ).splitlines()



    header_line = None



    for i, line in enumerate(lines):

        if "alpha" in line.lower() and "CL" in line:

            header_line = i

            break



    if header_line is None:

        raise ValueError(

            f"Could not find XFOIL polar header in {filename}"

        )



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

        raise ValueError(

            f"No numerical polar data found in {filename}"

        )



    df = pd.DataFrame(

        data,

        columns=["alpha", "cl", "cd"],

    )



    df = df.replace(

        [np.inf, -np.inf],

        np.nan,

    ).dropna(

        subset=["alpha", "cl", "cd"],

    )



    df = (

        df.sort_values("alpha")

        .drop_duplicates("alpha")

        .reset_index(drop=True)

    )



    return df





# ============================================================

# Diagnose polar coverage and basic data quality

# ============================================================



def diagnose_polar(xfoil_data, requested_aoa, stdout):

    """Compare requested and returned AoA points."""



    requested = np.asarray(requested_aoa, dtype=float)

    returned = np.round(

        xfoil_data["alpha"].to_numpy(dtype=float),

        3,

    )

    requested_rounded = np.round(requested, 3)



    missing = [

        float(alpha)

        for alpha in requested_rounded

        if not np.any(

            np.isclose(returned, alpha, atol=1e-3)

        )

    ]



    cl = xfoil_data["cl"].to_numpy()

    cd = xfoil_data["cd"].to_numpy()



    invalid_rows = (

        ~np.isfinite(cl)

        | ~np.isfinite(cd)

        | (cd < 0)

    )



    return {

        "missing_alpha": missing,

        "missing_count": len(missing),

        "invalid_row_count": int(np.sum(invalid_rows)),

        "solver_failure_message": has_convergence_failure(stdout),

        "all_requested_points_present": len(missing) == 0,

    }





# ============================================================

# Keep gaps in CL versus AoA plots

# ============================================================



def polar_with_gaps(xfoil_data, requested_aoa):

    """

    Map the polar onto the requested AoA grid.



    Missing requested angles become NaN so Matplotlib breaks the

    curve rather than joining across a missing section.

    """



    grid = pd.DataFrame({

        "alpha": np.asarray(requested_aoa, dtype=float)

    })



    data = xfoil_data.copy()



    grid["key"] = grid["alpha"].round(3)

    data["key"] = data["alpha"].round(3)



    merged = grid.merge(

        data[["key", "cl"]],

        on="key",

        how="left",

    )



    return (

        merged["alpha"].to_numpy(),

        merged["cl"].to_numpy(),

    )





# ============================================================

# Calculate RMSE against experimental data

# ============================================================



def calculate_errors(

    xfoil_data,

    experimental_data,

    requested_aoa,

):

    """

    Calculate CL errors at experimental AoA points.



    Do not extrapolate or interpolate across missing requested

    angles. The number of points used is returned with the metrics.

    """



    alpha_grid, cl_grid = polar_with_gaps(

        xfoil_data,

        requested_aoa,

    )



    alpha_exp = np.asarray(

        experimental_data["alpha"],

        dtype=float,

    )

    cl_exp = np.asarray(

        experimental_data["cl"],

        dtype=float,

    )



    alpha_valid = []

    cl_exp_valid = []

    cl_calc_valid = []



    for a_exp, cl_e in zip(alpha_exp, cl_exp):

        if a_exp < alpha_grid[0] or a_exp > alpha_grid[-1]:

            continue



        right = int(

            np.searchsorted(alpha_grid, a_exp, side="left")

        )



        # Exact grid match.

        if (

            right < len(alpha_grid)

            and np.isclose(alpha_grid[right], a_exp, atol=1e-8)

        ):

            if not np.isfinite(cl_grid[right]):

                continue



            cl_calc = cl_grid[right]



        # Interpolate only between adjacent, valid grid points.

        else:

            left = right - 1



            if left < 0 or right >= len(alpha_grid):

                continue



            if not (

                np.isfinite(cl_grid[left])

                and np.isfinite(cl_grid[right])

            ):

                continue



            if alpha_grid[right] - alpha_grid[left] > 0.51:

                continue



            cl_calc = np.interp(

                a_exp,

                [alpha_grid[left], alpha_grid[right]],

                [cl_grid[left], cl_grid[right]],

            )



        alpha_valid.append(a_exp)

        cl_exp_valid.append(cl_e)

        cl_calc_valid.append(cl_calc)



    if not alpha_valid:

        return {

            "rmse": np.nan,

            "mae": np.nan,

            "max_error": np.nan,

            "n_points": 0,

        }



    error = (

        np.asarray(cl_calc_valid)

        - np.asarray(cl_exp_valid)

    )



    return {

        "rmse": float(np.sqrt(np.mean(error**2))),

        "mae": float(np.mean(np.abs(error))),

        "max_error": float(np.max(np.abs(error))),

        "n_points": len(error),

    }





# ============================================================

# Plot CL versus AoA

# ============================================================



def plot_lift_curves(

    results,

    experimental_data,

    requested_aoa,

    title,

    plot_path,

):

    fig, ax = plt.subplots(figsize=(10, 6))



    ax.plot(

        experimental_data["alpha"],

        experimental_data["cl"],

        "ko",

        markersize=5,

        label="NASA Langley / Ladson experiment",

        zorder=5,

    )



    for result in results:

        alpha_plot, cl_plot = polar_with_gaps(

            result["xfoil"],

            requested_aoa,

        )



        rmse = result["error"]["rmse"]



        if np.isfinite(rmse):

            label = (

                f"{result['case']} "

                f"(RMSE={rmse:.4f}, "

                f"N={result['error']['n_points']})"

            )

        else:

            label = f"{result['case']} (RMSE=N/A)"



        ax.plot(

            alpha_plot,

            cl_plot,

            "-",

            linewidth=1.5,

            label=label,

        )



    ax.set_xlabel(r"Angle of attack, $\alpha$ [deg]")

    ax.set_ylabel(r"Lift coefficient, $C_L$")

    ax.set_title(title)

    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=8)



    fig.tight_layout()

    fig.savefig(plot_path, dpi=300)

    plt.show()

    plt.close(fig)



    print(f"Lift plot saved to: {plot_path}")





# ============================================================

# Single-angle XFOIL run for residual analysis

# ============================================================



def run_xfoil_single_alpha_residual(

    alpha,

    case_name,

    iterations,

    ncrit,

    panel_settings,

    output_dir,

):

    """

    Run one AoA in a fresh XFOIL process.



    Each console log therefore belongs to one requested AoA,

    making its residual history easier to interpret.

    """



    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)



    alpha_tag = f"a{alpha:+06.2f}".replace(".", "p")



    polar_name = f"{case_name}_{alpha_tag}.polar"

    input_path = output_dir / f"{case_name}_{alpha_tag}.inp"

    log_path = output_dir / f"{case_name}_{alpha_tag}.log"

    polar_path = output_dir / polar_name



    if polar_path.exists():

        polar_path.unlink()



    commands = build_xfoil_commands(

        polar_file=polar_name,

        aoa=[alpha],

        re=REYNOLDS_NUMBER,

        mach=MACH_NUMBER,

        iterations=iterations,

        ncrit=ncrit,

        panel_settings=panel_settings,

    )



    input_path.write_text(commands, newline="\r\n")



    try:

        process = subprocess.run(

            [XFOIL_EXE],

            input=commands,

            text=True,

            capture_output=True,

            cwd=output_dir,

            timeout=XFOIL_TIMEOUT_SECONDS,

        )



        stdout = process.stdout or ""

        stderr = process.stderr or ""



    except subprocess.TimeoutExpired as exc:

        stdout = exc.stdout or ""



        if isinstance(stdout, bytes):

            stdout = stdout.decode(errors="replace")



        log_path.write_text(stdout)



        return {

            "alpha": alpha,

            "history": pd.DataFrame(

                columns=["iteration", "rms"]

            ),

            "failed": True,

            "reason": "timeout",

            "log_path": str(log_path),

        }



    log_path.write_text(

        stdout + "\n\n--- STDERR ---\n" + stderr

    )



    history = parse_residual_history(stdout)



    failed = (

        has_convergence_failure(stdout)

        or process.returncode != 0

    )



    if history.empty:

        reason = "no residuals parsed"

    elif failed:

        reason = "explicit solver failure or non-zero return code"

    else:

        reason = "no explicit failure detected"



    return {

        "alpha": alpha,

        "history": history,

        "failed": failed,

        "reason": reason,

        "log_path": str(log_path),

    }





# ============================================================

# Plot iteration versus residual for all AoA cases

# ============================================================



def plot_study_residuals(
    study_name,
    description,
    cases,
    aoa_values=RESIDUAL_AOA,
):
    """
    Plot RMS residual against iteration number.

    For the iteration study, plot only the 200-iteration case and
    label each AoA curve with the last iteration recorded.
    Other studies retain one subplot per parameter case.
    """
    output_dir = Path(
        get_path("results_validation", description=description)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    residual_dir = output_dir / "single_alpha_logs"
    residual_dir.mkdir(parents=True, exist_ok=True)

    is_iteration_study = study_name == "iterations"
    n_cases = len(cases)
    fig, axes = plt.subplots(
        n_cases, 1,
        figsize=(11, max(5, 3.4 * n_cases)),
        squeeze=False,
    )
    axes = axes.ravel()

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(
        vmin=float(np.min(aoa_values)),
        vmax=float(np.max(aoa_values)),
    )
    diagnostic_rows = []

    for ax, case in zip(axes, cases):
        print(f"\\nResidual diagnostics: {case['case']}")

        for alpha in aoa_values:
            run = run_xfoil_single_alpha_residual(
                alpha=float(alpha),
                case_name=case["case"],
                iterations=case["iterations"],
                ncrit=case["ncrit"],
                panel_settings=case["panel_settings"],
                output_dir=residual_dir,
            )
            history = run["history"]
            last_iteration = (
                int(history["iteration"].max())
                if not history.empty else np.nan
            )

            diagnostic_rows.append({
                "study": study_name,
                "case": case["case"],
                "iterations_limit": case["iterations"],
                "ncrit": case["ncrit"],
                "panel_settings": str(case["panel_settings"]),
                "alpha": alpha,
                "residual_records": len(history),
                "last_iteration_recorded": last_iteration,
                "initial_rms": (
                    float(history["rms"].iloc[0])
                    if not history.empty else np.nan
                ),
                "final_rms": (
                    float(history["rms"].iloc[-1])
                    if not history.empty else np.nan
                ),
                "failed": run["failed"],
                "reason": run["reason"],
                "log_file": run["log_path"],
            })

            if history.empty:
                print(
                    f"  AoA={alpha:5.1f}: no residual history parsed; "
                    f"inspect {run['log_path']}"
                )
                continue

            label = (
                f"AoA = {alpha:g}° ({int(last_iteration)} iterations)"
                if is_iteration_study else None
            )
            ax.semilogy(
                history["iteration"],
                history["rms"],
                color=cmap(norm(alpha)),
                linewidth=1.2,
                alpha=0.85,
                label=label,
            )

        ax.set_title(
            f"{case['case']} | ITER={case['iterations']}, "
            f"Ncrit={case['ncrit']}, "
            f"panels={case['panel_settings'] or 'default'}",
            fontsize=9,
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMS residual")
        ax.grid(True, which="both", alpha=0.25)

        if is_iteration_study:
            ax.legend(
                title="AoA (iterations to converge)",
                fontsize=8,
                title_fontsize=9,
                loc="best",
            )

    if not is_iteration_study:
        colour_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colour_map.set_array([])
        fig.colorbar(
            colour_map,
            ax=list(axes),
            label="Angle of attack [deg]",
            pad=0.02,
        )

    fig.suptitle(
        f"NACA 0012 — {study_name}: residual convergence by AoA",
        y=1.002,
    )
    fig.tight_layout()

    plot_path = output_dir / f"{study_name}_residuals_by_alpha.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    diagnostics = pd.DataFrame(diagnostic_rows)
    csv_path = output_dir / f"{study_name}_residual_diagnostics.csv"
    diagnostics.to_csv(csv_path, index=False)

    print(f"Residual plot saved to: {plot_path}")
    print(f"Residual diagnostics saved to: {csv_path}")
    return diagnostics


def run_study(

    study_name,

    description,

    cases,

    experimental_data,

):

    output_dir = Path(

        get_path(

            "results_validation",

            description=description,

        )

    )

    output_dir.mkdir(parents=True, exist_ok=True)



    results = []



    for case in cases:

        run = run_xfoil(

            case_name=case["case"],

            aoa=AOA,

            re=REYNOLDS_NUMBER,

            mach=MACH_NUMBER,

            iterations=case["iterations"],

            ncrit=case["ncrit"],

            output_dir=output_dir,

            panel_settings=case["panel_settings"],

        )



        xfoil_data = read_xfoil_polar(run["polar_path"])



        diagnostics = diagnose_polar(

            xfoil_data,

            AOA,

            run["stdout"],

        )



        error = calculate_errors(

            xfoil_data,

            experimental_data,

            AOA,

        )



        result = {

            **case,

            "run": run,

            "xfoil": xfoil_data,

            "diagnostics": diagnostics,

            "error": error,

        }

        results.append(result)



        print(

            f"  RMSE={error['rmse']:.5f} "

            f"(experimental points={error['n_points']}), "

            f"missing AoA={diagnostics['missing_count']}, "

            f"solver warning={diagnostics['solver_failure_message']}"

        )



        if diagnostics["missing_alpha"]:

            print(

                f"  Missing AoA: {diagnostics['missing_alpha']}"

            )



    summary_rows = []



    for result in results:

        error = result["error"]

        diag = result["diagnostics"]



        summary_rows.append({

            "study": study_name,

            "case": result["case"],

            "iterations": result["iterations"],

            "ncrit": result["ncrit"],

            "panel_settings": str(result["panel_settings"]),

            "rmse_cl": error["rmse"],

            "mae_cl": error["mae"],

            "max_abs_error_cl": error["max_error"],

            "n_experimental_points": error["n_points"],

            "missing_aoa_count": diag["missing_count"],

            "missing_aoa_deg": ", ".join(

                f"{alpha:g}"

                for alpha in diag["missing_alpha"]

            ),

            "solver_failure_message": (

                diag["solver_failure_message"]

            ),

            "invalid_polar_rows": diag["invalid_row_count"],

            "all_requested_points_present": (

                diag["all_requested_points_present"]

            ),

            "return_code": result["run"]["return_code"],

            "polar_file": result["run"]["polar_path"],

            "log_file": result["run"]["log_path"],

        })



    summary = pd.DataFrame(summary_rows)



    summary_path = output_dir / f"{study_name}_summary.csv"

    summary.to_csv(summary_path, index=False)



    plot_lift_curves(

        results=results,

        experimental_data=experimental_data,

        requested_aoa=AOA,

        title=f"NACA 0012 — {study_name}: XFOIL vs Ladson",

        plot_path=output_dir / f"{study_name}_cl_vs_alpha.png",

    )



    print(f"Summary CSV saved to: {summary_path}")



    return results, summary





# ============================================================

# Main: three isolated parameter studies

# ============================================================



def main():

    if not os.path.isfile(XFOIL_EXE):

        raise FileNotFoundError(

            f"XFOIL executable not found: {XFOIL_EXE}"

        )



    experimental_data = EXPERIMENTAL_DATA[REYNOLDS_NUMBER]



    # --------------------------------------------------------

    # 1. Iteration sensitivity

    # Hold Ncrit and panel distribution constant.

    # --------------------------------------------------------



    iteration_cases = [

        {

            "case": f"iterations_{n}",

            "iterations": n,

            "ncrit": BASE_NCRIT,

            "panel_settings": None,

        }

        for n in ITERATION_VALUES

    ]



    iteration_results, iteration_summary = run_study(

        study_name="iterations",

        description=STUDY_DESCRIPTIONS["iterations"],

        cases=iteration_cases,

        experimental_data=experimental_data,

    )



    plot_study_residuals(

        study_name="iterations",

        description=STUDY_DESCRIPTIONS["iterations"],

        cases=[iteration_cases[-1]],  # ITER=200 only

    )



    # --------------------------------------------------------

    # 2. Ncrit sensitivity

    # Hold iteration limit and panel distribution constant.

    # --------------------------------------------------------



    ncrit_cases = [

        {

            "case": f"ncrit_{n}",

            "iterations": BASE_ITERATIONS,

            "ncrit": n,

            "panel_settings": None,

        }

        for n in NCRIT_VALUES

    ]



    ncrit_results, ncrit_summary = run_study(

        study_name="ncrit",

        description=STUDY_DESCRIPTIONS["ncrit"],

        cases=ncrit_cases,

        experimental_data=experimental_data,

    )



    plot_study_residuals(

        study_name="ncrit",

        description=STUDY_DESCRIPTIONS["ncrit"],

        cases=ncrit_cases,

    )



    # --------------------------------------------------------

    # 3. Panel sensitivity

    # Hold iteration limit and Ncrit constant.

    # --------------------------------------------------------



    panel_cases = [

        {

            "case": f"panels_{name}",

            "iterations": BASE_ITERATIONS,

            "ncrit": BASE_NCRIT,

            "panel_settings": settings,

        }

        for name, settings in PANEL_CASES.items()

    ]



    panel_results, panel_summary = run_study(

        study_name="panels",

        description=STUDY_DESCRIPTIONS["panels"],

        cases=panel_cases,

        experimental_data=experimental_data,

    )



    plot_study_residuals(

        study_name="panels",

        description=STUDY_DESCRIPTIONS["panels"],

        cases=panel_cases,

    )



    # --------------------------------------------------------

    # Combined summary CSV

    # --------------------------------------------------------



    combined = pd.concat(

        [

            iteration_summary,

            ncrit_summary,

            panel_summary,

        ],

        ignore_index=True,

    )



    combined_dir = Path(

        get_path(

            "results_validation",

            description="08_combined_summary",

        )

    )

    combined_dir.mkdir(parents=True, exist_ok=True)



    combined_path = (

        combined_dir / "all_validation_results.csv"

    )

    combined.to_csv(combined_path, index=False)



    print("\n" + "=" * 80)

    print("COMBINED XFOIL VALIDATION SUMMARY")

    print("=" * 80)



    print(

        combined[

            [

                "study",

                "case",

                "rmse_cl",

                "n_experimental_points",

                "missing_aoa_count",

                "solver_failure_message",

            ]

        ].to_string(index=False)

    )



    print(f"\nCombined CSV saved to: {combined_path}")





if __name__ == "__main__":

    main()
