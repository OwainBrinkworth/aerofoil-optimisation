import os
import glob
import re

from defaults import params
from paths import get_path


def extract_cl_values(results_df, pattern="polar_*.txt", test=False, *, large_file_threshold=1_000_000, tail_bytes=8192, verbose=False):
    """
    Extract CL values from multiple XFOIL polar files.

    Parameters:
        results_dir (str): directory containing polar files
        pattern (str): file pattern (default: polar_*.txt)

    Returns:
        list: CL values (None if missing)
    """
    test_type = 'test' if test else 'experiment'
    n_samples = params[test_type]['n_samples']
    results_dir = get_path(f'results_solver', test=test)

    file_paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    length = len(results_df)
    cl_values = [None] * length

    for i, file in enumerate(file_paths):
        cl = None
        try:
            # Quick progress
            if verbose and i and i % 100 == 0:
                print(f"Parsing polar files: {i}/{len(file_paths)}")

            size = os.path.getsize(file)

            # Read only tail for very large files to avoid long blocking reads
            if size > large_file_threshold:
                with open(file, "rb") as f:
                    if size > tail_bytes:
                        f.seek(-tail_bytes, os.SEEK_END)
                    raw = f.read()
                text = raw.decode("utf-8", errors="ignore")
                lines = text.splitlines()
            else:
                with open(file, "r", errors="ignore") as f:
                    lines = f.readlines()

            # First try to parse numeric lines like: <alpha> <CL> <...>
            angle_cl_map = {}
            last_angle_key = None
            float_line_re = re.compile(r"^\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)? )")
            # Relaxed regex: match two floats at start of line (allow additional columns)
            float_line_re = re.compile(r"^\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)")
            for line in lines:
                m = float_line_re.match(line)
                if m:
                    try:
                        angle = float(m.group(1))
                        cl_val = float(m.group(2))
                        angle_cl_map[angle] = cl_val
                        last_angle_key = angle
                    except Exception:
                        continue

            if angle_cl_map:
                # use the last occurrence parsed (preserves order of appearance)
                if last_angle_key is not None:
                    cl = angle_cl_map[last_angle_key]
            else:
                # Fallback: search for 'CL = <value>' occurrences in the text
                joined = "\n".join(lines) if isinstance(lines, list) else lines
                matches = re.findall(r"CL\s*=\s*([0-9.+\-E]+)", joined)
                if matches:
                    try:
                        cl = float(matches[-1])
                    except Exception:
                        cl = None

        except Exception as e:
            print(f"Error reading {file}: {e}")
            cl = None

        if cl is None:
            print(f"Warning: No CL found in {file}")

        # Determine index safely
        try:
            index_str = os.path.basename(file).split('polar_')[-1].split('.txt')[0]
            idx = int(index_str)
        except Exception:
            print(f"Warning: Unable to determine index from filename {file}")
            continue

        if 0 <= idx < length:
            cl_values[idx] = cl
        else:
            # Out-of-range index (maybe different sampling); skip but warn
            print(f"Warning: index {idx} out of range for results dataframe (len={length})")

    results_df["CL"] = cl_values
    return results_df
