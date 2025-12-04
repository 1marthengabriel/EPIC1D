#!/usr/bin/env python3
"""
scan_epic1d.py

Parallel parameter-scan driver for epic1d.py
PIC Lab — University of York

Features:
- Multiprocessing + tqdm progress bars
- Master logging to results/scan_runs/logs/scan_log.txt
- Aggregated CSV saved to results/scan_runs/csv/scan_results.csv
- Plots (with error bars) saved to results/scan_runs/plots/
  * metric vs Np (curves per Nc) for each L
  * metric vs Nc (curves per Np) for each L
  * metric vs L  (curves per Np) for each Nc
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Import PIC solver (assumes epic1d.py in same package / path)
from epic1d import landau, Summary, run

# ---------- Directories ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SCAN_DIR = os.path.join(RESULTS_DIR, "scan_runs")
CSV_DIR = os.path.join(SCAN_DIR, "csv")
PLOT_DIR = os.path.join(SCAN_DIR, "plots")
LOG_DIR = os.path.join(SCAN_DIR, "logs")

for d in [RESULTS_DIR, SCAN_DIR, CSV_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- Logging ----------
LOG_PATH = os.path.join(LOG_DIR, "scan_log.txt")
LOG_FILE = open(LOG_PATH, "a")

def log(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    LOG_FILE.write(line + "\n")
    LOG_FILE.flush()
    print(line)

# ---------- Single simulation runner ----------
def run_single_sim(args):
    """
    Run a single repeat of the simulation for (Np, Nc, L)
    Returns: runtime, noise_lvl, omega, gamma
    """
    Np, Nc, L = args
    try:
        # Initial condition
        pos, vel = landau(int(Np), float(L))
        summary = Summary()

        # Time only the run() call
        t0 = time.time()
        run(pos, vel, float(L), int(Nc),
            out=[summary],
            output_times=np.linspace(0.0, 20.0, 50))
        runtime = time.time() - t0

        t = np.array(summary.t)
        amp = np.array(summary.firstharmonic)

        # Peak detection
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(amp)
        if len(peaks) == 0:
            return (runtime, np.nan, np.nan, np.nan)

        peak_times = t[peaks]
        peak_vals = amp[peaks]

        # Find where noise dominates (first increase)
        noise_start = None
        for i in range(1, len(peak_vals)):
            if peak_vals[i] > peak_vals[i - 1]:
                noise_start = i
                break

        if noise_start is None:
            signal = peak_vals
            signal_times = peak_times
            noise = np.array([])
        else:
            signal = peak_vals[:noise_start]
            signal_times = peak_times[:noise_start]
            noise = peak_vals[noise_start:]

        # Noise RMS
        noise_lvl = np.sqrt(np.mean(noise**2)) if noise.size > 0 else np.nan

        # Frequency from signal peak spacing
        if len(signal_times) > 2:
            dt = np.diff(signal_times)
            mean_dt = np.mean(dt)
            omega = 2.0 * np.pi / mean_dt if mean_dt != 0 else np.nan
        else:
            omega = np.nan

        # Damping via log-linear fit
        if len(signal) > 2 and np.all(signal > 0):
            slope, intercept = np.polyfit(signal_times, np.log(signal), 1)
            gamma = -slope
        else:
            gamma = np.nan

        return (runtime, noise_lvl, omega, gamma)

    except Exception as e:
        # In case of crash, log and return nans
        log(f"ERROR in run_single_sim (Np={Np},Nc={Nc},L={L}): {repr(e)}")
        return (np.nan, np.nan, np.nan, np.nan)

# ---------- Repeat wrapper (parallel repeats) ----------
def run_case(Np, Nc, L, repeats=5):
    """
    Run `repeats` independent runs for a single (Np, Nc, L) case.
    Uses multiprocessing to run repeats in parallel.
    Returns:
      mean (runtime, noise, omega, gamma),
      sem  (runtime_err, noise_err, omega_err, gamma_err)
    """
    args_list = [(Np, Nc, L) for _ in range(repeats)]
    nprocs = min(cpu_count(), repeats)

    # Use Pool in context manager
    with Pool(processes=nprocs) as pool:
        # imap is lazy; wrap with tqdm for progress of repeats
        results = list(tqdm(pool.imap(run_single_sim, args_list),
                            total=repeats,
                            desc=f"Repeats (Np={Np},Nc={Nc},L={L:.2f})",
                            leave=False))

    results = np.array(results, dtype=float)  # shape (repeats, 4)
    # Calculate mean and SEM (ignoring nan)
    mean = np.nanmean(results, axis=0)
    # sem: std / sqrt(n_eff) where n_eff counts non-nan items per column
    sem = []
    for col in range(results.shape[1]):
        colvals = results[:, col]
        valid = ~np.isnan(colvals)
        if np.sum(valid) > 1:
            sem_val = np.nanstd(colvals[valid], ddof=1) / np.sqrt(np.sum(valid))
        else:
            sem_val = np.nan
        sem.append(sem_val)
    sem = np.array(sem, dtype=float)
    
    return mean, sem


# ---------- Main parameter scan ----------
def parameter_scan(particle_list=None, cell_list=None, length_list=None, repeats=5):
    # Default scan (can be overridden)
    if particle_list is None:
        particle_list = [500, 1000, 5000, 20000]
    if cell_list is None:
        cell_list = [10, 20, 40, 80]
    if length_list is None:
        length_list = [4*np.pi, 6*np.pi, 8*np.pi, 10*np.pi]

    log("===== Parameter scan STARTED =====")
    log(f"Particle_list: {particle_list}")
    log(f"Cell_list: {cell_list}")
    log(f"Length_list: {length_list}")
    log(f"Repeats per case: {repeats}")
    start_time = time.time()

    csv_path = os.path.join(CSV_DIR, "scan_results.csv")
    with open(csv_path, "w") as fh:
        fh.write("Np,Nc,L,"
                 "runtime,runtime_err,"
                 "noise,noise_err,"
                 "omega,omega_err,"
                 "gamma,gamma_err\n")

        total_cases = len(particle_list) * len(cell_list) * len(length_list)
        pbar = tqdm(total=total_cases, desc="Overall scan", leave=True)

        for Np in particle_list:
            for Nc in cell_list:
                for L in length_list:
                    log(f"Starting case: Np={Np}, Nc={Nc}, L={L:.4f}")
                    mean, sem = run_case(Np, Nc, L, repeats=repeats)

                    runtime, noise, omega, gamma = mean
                    runtime_err, noise_err, omega_err, gamma_err = sem

                    fh.write(f"{int(Np)},{int(Nc)},{float(L):.6f},"
                             f"{float(runtime):.6e},{float(runtime_err):.6e},"
                             f"{float(noise):.6e},{float(noise_err):.6e},"
                             f"{float(omega):.6e},{float(omega_err):.6e},"
                             f"{float(gamma):.6e},{float(gamma_err):.6e}\n")

                    log(f"Finished case: Np={Np}, Nc={Nc}, L={L:.4f} | "
                        f"runtime={runtime:.3f}s ± {runtime_err:.3f}, "
                        f"noise={noise:.3e} ± {noise_err:.3e}, "
                        f"omega={omega:.3f} ± {omega_err:.3f}, "
                        f"gamma={gamma:.3f} ± {gamma_err:.3f}")

                    pbar.update(1)

        pbar.close()

    elapsed = time.time() - start_time
    log(f"===== Parameter scan FINISHED in {elapsed/60.0:.2f} minutes =====")
    log(f"Results CSV: {csv_path}")
    return csv_path


# ---------- Plotting utilities ----------
def _plot_metric_vs_x(df, xcol, metric, metric_err, groupby, title, outpath, xlabel):
    """
    Generic: plot metric vs xcol, with separate curves per groupby value.
    df: pandas.DataFrame containing columns
    xcol: column name for x axis (e.g., 'Np' or 'Nc' or 'L')
    metric: column name for y (e.g., 'gamma')
    metric_err: column name for error (e.g., 'gamma_err')
    groupby: column name to split curves (e.g., 'Nc' or 'Np')
    """
    import pandas as pd
    unique_groups = sorted(df[groupby].unique())
    plt.figure(figsize=(7,5))
    for g in unique_groups:
        sub = df[df[groupby] == g].sort_values(by=xcol)
        if sub.empty:
            continue
        x = sub[xcol].values
        y = sub[metric].values
        yerr = sub[metric_err].values
        # Use errorbar
        plt.errorbar(x, y, yerr=yerr, marker='o', linestyle='-', label=f"{groupby}={g}")
    plt.xlabel(xlabel)
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def generate_plots(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)

    # convenience: ensure numeric types
    df['Np'] = df['Np'].astype(float)
    df['Nc'] = df['Nc'].astype(float)
    df['L']  = df['L'].astype(float)

    # Calculate overall statistics from the dataframe
    runtime_mean = df['runtime'].mean()
    runtime_err_mean = df['runtime_err'].mean()
    noise_mean = df['noise'].mean()
    noise_err_mean = df['noise_err'].mean()
    omega_mean = df['omega'].mean()
    omega_err_mean = df['omega_err'].mean()
    gamma_mean = df['gamma'].mean()
    gamma_err_mean = df['gamma_err'].mean()

    print(f"Overall Runtime = {runtime_mean:.3f} ± {runtime_err_mean:.3f}")
    print(f"Overall Noise = {noise_mean:.3e} ± {noise_err_mean:.3e}")
    print(f"Overall Omega = {omega_mean:.3f} ± {omega_err_mean:.3f}")
    print(f"Overall Gamma = {gamma_mean:.3f} ± {gamma_err_mean:.3f}")

    metrics = [
        ("runtime", "runtime_err"),
        ("noise", "noise_err"),
        ("omega", "omega_err"),
        ("gamma", "gamma_err")
    ]


    # 1) For each L: metric vs Np (curves per Nc)
    for Lval in sorted(df['L'].unique()):
        subL = df[df['L'] == Lval]
        for metric, metric_err in metrics:
            title = f"{metric} vs Np (L={Lval:.3f}), curves per Nc"
            outname = f"{metric}_vs_Np_L_{Lval:.3f}.png".replace(".", "p")
            outpath = os.path.join(PLOT_DIR, outname)
            _plot_metric_vs_x(subL, "Np", metric, metric_err, "Nc", title, outpath, xlabel="Np")

    # 2) For each L: metric vs Nc (curves per Np)
    for Lval in sorted(df['L'].unique()):
        subL = df[df['L'] == Lval]
        for metric, metric_err in metrics:
            title = f"{metric} vs Nc (L={Lval:.3f}), curves per Np"
            outname = f"{metric}_vs_Nc_L_{Lval:.3f}.png".replace(".", "p")
            outpath = os.path.join(PLOT_DIR, outname)
            _plot_metric_vs_x(subL, "Nc", metric, metric_err, "Np", title, outpath, xlabel="Nc")

    # 3) For each Nc: metric vs L (curves per Np)
    for Ncval in sorted(df['Nc'].unique()):
        subNc = df[df['Nc'] == Ncval]
        for metric, metric_err in metrics:
            title = f"{metric} vs L (Nc={Ncval:.0f}), curves per Np"
            outname = f"{metric}_vs_L_Nc_{int(Ncval)}.png".replace(".", "p")
            outpath = os.path.join(PLOT_DIR, outname)
            _plot_metric_vs_x(subNc, "L", metric, metric_err, "Np", title, outpath, xlabel="L")

    log("All plots generated and saved to: " + PLOT_DIR)


# ---------- Main ----------
if __name__ == "__main__":
    # Small quick test run settings (edit here for full scan)
    particle_list = [500, 1000, 5000, 20000]   # change as needed
    cell_list     = [10, 20, 40, 80]           # change as needed
    length_list   = [4*np.pi, 6*np.pi, 8*np.pi, 10*np.pi]
    repeats = 5

    csv = parameter_scan(particle_list, cell_list, length_list, repeats=repeats)
    generate_plots(csv)

    LOG_FILE.close()
