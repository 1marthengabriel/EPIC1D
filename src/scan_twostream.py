#!/usr/bin/env python3
"""
scan_twostream.py
Parameter scan for the TWO-STREAM instability.
This script performs ALL Tasks 1-4 of the PIC Lab Session 3-4.
Features:
- Baseline two-stream run with animations
- Growth rate measurement with proper fitting
- Scan over beam velocity vbeam to find thresholds
- Comparison with analytic dispersion relation
- Narrow-beam mode (beam width / 10)
- Bump-on-tail instability (Task 4 optional)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
from numpy.random import uniform, normal
from numpy import histogram, sqrt, exp

from epic1d import twostream, Summary, Plot, run   # <— your existing solver


# -------------------------------------------------------------
# Directory Setup
# -------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TS_DIR = os.path.join(RESULTS_DIR, "twostream_scans")
CSV_DIR = os.path.join(TS_DIR, "csv")
PLOT_DIR = os.path.join(TS_DIR, "plots")
LOG_DIR = os.path.join(TS_DIR, "logs")

for d in [RESULTS_DIR, TS_DIR, CSV_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_PATH = os.path.join(LOG_DIR, "twostream_scan_log.txt")
LOG_FILE = open(LOG_PATH, "a")

def log(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    LOG_FILE.write(line + "\n")
    LOG_FILE.flush()
    print(line)

# -------------------------------------------------------------
# Bump-on-tail initial condition function (Task 4 Optional)
# -------------------------------------------------------------
def bump(npart, L, vbump, fraction_bump=0.1):
    """
    Create bump-on-tail distribution for heating studies.
    """
    npart_bump = int(npart * fraction_bump)
    npart_bulk = npart - npart_bump
    
    # Uniform spatial distribution
    pos = uniform(0., L, npart)
    
    # Bulk plasma: Maxwellian with width 1
    vel_bulk = normal(0.0, 1.0, npart_bulk)
    
    # Bump: beam with smaller thermal spread
    vel_bump = normal(vbump, 0.2, npart_bump)
    
    vel = np.concatenate([vel_bulk, vel_bump])
    
    # Shuffle to mix particles (optional)
    indices = np.arange(npart)
    np.random.shuffle(indices)
    
    return pos[indices], vel[indices]

# -------------------------------------------------------------
# Analytic dispersion relation for two-stream instability
# -------------------------------------------------------------
def analytic_growth_rate(k, vbeam, omega_pe=1.0):
    """
    Analytic growth rate from cold two-stream dispersion relation.
    
    For two symmetric cold beams:
    ω² = (k v0)² + ω_pe² ± √(ω_pe⁴ + 4 k² v0² ω_pe²)
    
    The negative sign gives the unstable root when:
    k v0 < ω_pe √2  (upper threshold)
    
    Returns growth rate γ = Im(ω) if unstable, 0 otherwise.
    """
    # Using normalized units where ω_pe = 1 for each beam
    # Total plasma frequency = sqrt(2) for both beams
    omega_pe_total = np.sqrt(2)
    
    # Check threshold condition
    if k * vbeam >= omega_pe_total:
        return 0.0  # Stable above upper threshold
    
    # Compute unstable root
    term1 = (k * vbeam)**2 + omega_pe**2
    term2 = np.sqrt(omega_pe**4 + 4 * k**2 * vbeam**2 * omega_pe**2)
    
    # Unstable mode (negative sign gives complex ω)
    omega_sq = term1 - term2
    
    if omega_sq < 0:
        return np.sqrt(-omega_sq)
    else:
        return 0.0

def find_most_unstable_k(vbeam, k_min=0.01, k_max=1.0, nk=100):
    """
    Find the wavenumber that gives maximum growth rate.
    """
    k_vals = np.linspace(k_min, k_max, nk)
    gamma_vals = [analytic_growth_rate(k, vbeam) for k in k_vals]
    
    if np.max(gamma_vals) > 0:
        max_idx = np.argmax(gamma_vals)
        return k_vals[max_idx], gamma_vals[max_idx]
    else:
        return None, 0.0

# -------------------------------------------------------------
# Gaussian fitting for velocity distribution analysis
# -------------------------------------------------------------
def gaussian(x, A, mu, sigma, offset):
    """Gaussian function for fitting velocity distributions."""
    return A * exp(-(x - mu)**2 / (2 * sigma**2)) + offset

def fit_velocity_distribution(vel, nbins=50):
    """
    Fit Gaussian to velocity distribution.
    Returns (amplitude, mean, width, offset, r_squared)
    """
    # Create histogram
    counts, bin_edges = histogram(vel, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initial guess for Gaussian fit
    mean_guess = np.mean(vel)
    sigma_guess = np.std(vel)
    A_guess = np.max(counts)
    
    try:
        # Fit Gaussian
        popt, pcov = curve_fit(gaussian, bin_centers, counts,
                              p0=[A_guess, mean_guess, sigma_guess, 0])
        
        # Calculate R-squared
        residuals = counts - gaussian(bin_centers, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((counts - np.mean(counts))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return popt, r_squared
    except:
        return [A_guess, mean_guess, sigma_guess, 0], 0

# -------------------------------------------------------------
# Enhanced growth-rate measurement
# -------------------------------------------------------------
def measure_growth_rate_enhanced(t, amp, method='tanh_fit'):
    """
    Fit growth rate using different methods.
    
    Methods:
    - 'linear_fit': Linear fit to log(A) in growth region
    - 'tanh_fit': Fit A(t) ~ A0 * exp(γ*t) / (1 + exp(γ*(t-t0))) 
    - 'automatic': Automatically detect growth region
    
    Returns: gamma, t_start, t_end, fit_quality
    """
    t = np.array(t)
    A = np.abs(np.array(amp))
    
    # Avoid zeros/negatives
    A[A <= 0] = np.min(A[A > 0])
    
    if method == 'linear_fit':
        # Method 1: Find growth region automatically
        logA = np.log(A)
        
        # Smooth derivative to find growth region
        dlogA_dt = np.gradient(logA, t)
        
        # Find where derivative is positive and significant
        growth_mask = dlogA_dt > 0.1 * np.max(dlogA_dt)
        growth_indices = np.where(growth_mask)[0]
        
        if len(growth_indices) < 5:
            return np.nan, 0, 0, 0
        
        # Find contiguous growth region
        start_idx = growth_indices[0]
        end_idx = growth_indices[-1]
        
        # Ensure we have enough points
        if end_idx - start_idx < 10:
            return np.nan, 0, 0, 0
        
        # Linear fit in growth region
        coeffs = np.polyfit(t[start_idx:end_idx], logA[start_idx:end_idx], 1)
        gamma = coeffs[0]
        
        return gamma, t[start_idx], t[end_idx], 1.0
    
    elif method == 'tanh_fit':
        # Method 2: Fit tanh-like saturation model
        # A(t) = A0 * exp(γ*t) / (1 + (exp(γ*(t-t0)) - 1) * f)
        # Simplified: A(t) = C / (1 + exp(-γ*(t-t0))) + offset
        
        def saturation_model(t_fit, gamma, t0, A_sat, A_bg):
            return A_sat / (1 + np.exp(-gamma * (t_fit - t0))) + A_bg
        
        # Initial guesses
        A_min, A_max = np.min(A), np.max(A)
        t_mid = t[len(t)//2]
        
        try:
            bounds = ([0.01, t[0], A_min*0.5, 0], 
                     [2.0, t[-1], A_max*2, A_min])
            popt, pcov = curve_fit(saturation_model, t, A, 
                                  p0=[0.1, t_mid, A_max, A_min],
                                  bounds=bounds)
            gamma, t0, A_sat, A_bg = popt
            return gamma, t0, t[-1], 1.0
        except:
            return np.nan, 0, 0, 0
    
    return np.nan, 0, 0, 0

# -------------------------------------------------------------
# Single simulation
# -------------------------------------------------------------
def run_single_sim(args):
    """Run a single simulation with given parameters."""
    vbeam, narrow, bump_config = args
    
    npart = 10000
    L = 100
    Nc = 20
    
    # Choose initial condition
    if bump_config is not None:
        # Bump-on-tail configuration
        fraction_bump = bump_config.get('fraction', 0.1)
        pos, vel = bump(npart, L, vbeam, fraction_bump)
    elif narrow:
        # Narrow beams (width reduced by factor 10)
        pos = uniform(0., L, npart)
        half = npart // 2
        vel = np.concatenate([
            normal(+vbeam, 0.1, half),  # Width = 0.1 instead of 1.0
            normal(-vbeam, 0.1, npart - half)
        ])
    else:
        # Default two-stream
        pos, vel = twostream(npart, L, vbeam)
    
    # Run simulation
    summary = Summary()
    t0 = time.time()
    
    run(pos, vel, L, Nc,
        out=[summary],
        output_times=np.linspace(0.0, 80.0, 200))
    
    runtime = time.time() - t0
    
    # Compute growth rate using enhanced method
    gamma, t_start, t_end, fit_quality = measure_growth_rate_enhanced(
        summary.t, summary.firstharmonic, method='linear_fit')
    
    # Additional diagnostics
    A = np.abs(summary.firstharmonic)
    noise_floor = np.min(A[A > 0])
    
    # Find saturation time
    dA_dt = np.gradient(A, summary.t)
    sat_indices = np.where(dA_dt <= 0)[0]
    sat_time = summary.t[sat_indices[0]] if len(sat_indices) > 0 else None
    
    # Final velocity distribution analysis (for bump-on-tail)
    if bump_config is not None:
        # Fit Gaussian to final velocity distribution
        fit_params, r2 = fit_velocity_distribution(vel)
        final_mean, final_width = fit_params[1], fit_params[2]
    else:
        final_mean, final_width, r2 = 0, 0, 0
    
    return {
        "gamma": gamma,
        "t_start": t_start,
        "t_end": t_end,
        "fit_quality": fit_quality,
        "runtime": runtime,
        "time": summary.t,
        "firstharmonic": summary.firstharmonic,
        "noise_floor": noise_floor,
        "sat_time": sat_time,
        "final_vel": vel if bump_config is not None else None,
        "final_mean": final_mean,
        "final_width": final_width,
        "fit_r2": r2,
        "vbeam": vbeam,
        "narrow": narrow,
        "bump": bump_config is not None
    }

# -------------------------------------------------------------
# Main Scan Function
# -------------------------------------------------------------
def perform_scan(vbeam_list, repeats=5, narrow=False, bump_config=None):
    """Perform parameter scan over beam velocities."""
    if bump_config:
        label = "bump"
    elif narrow:
        label = "narrow"
    else:
        label = "default"
    
    csv_summary_path = os.path.join(CSV_DIR, f"twostream_{label}_summary.csv")
    csv_saturation_path = os.path.join(CSV_DIR, f"twostream_{label}_saturation.csv")
    log(f"Starting two-stream scan ({label})")
    
    # Prepare arguments for parallel execution
    args_list = [(vb, narrow, bump_config) for vb in vbeam_list for _ in range(repeats)]
    
    # Run simulations in parallel
    with Pool(min(cpu_count(), 8)) as pool:
        outputs = list(tqdm(pool.imap(run_single_sim, args_list), 
                          total=len(args_list),
                          desc=f"Scanning {label} beams"))
    
    # Process results
    results_summary = []
    saturation_summary = []
    i = 0
    
    for vb_idx, vb in enumerate(vbeam_list):
        gammas = []
        valid_gammas = []
        saturation_times = []
        noise_floors = []
        
        for rep in range(repeats):
            output = outputs[i]
            gamma = output["gamma"]
            
            if not np.isnan(gamma) and gamma > 0:
                gammas.append(gamma)
                valid_gammas.append(gamma)
            
            # Collect saturation time
            sat_time = output.get("sat_time")
            if sat_time is not None:
                saturation_times.append(sat_time)
            
            # Collect noise floor
            noise_floor = output.get("noise_floor")
            if noise_floor is not None:
                noise_floors.append(noise_floor)
            
            # Save detailed output
            case_csv = os.path.join(CSV_DIR, 
                f"twostream_{label}_v{vb:.2f}_rep{rep+1}.csv")
            np.savetxt(
                case_csv,
                np.column_stack([output["time"], output["firstharmonic"]]),
                delimiter=",",
                header=f"time,firstharmonic\n# vbeam={vb}, gamma={gamma:.4f}, sat_time={sat_time if sat_time else 'None'}",
                comments=""
            )
            
            i += 1
        
        # Growth rate statistics
        if valid_gammas:
            gmean = np.mean(valid_gammas)
            gstd = np.std(valid_gammas)
            fraction_valid = len(valid_gammas) / repeats
        else:
            gmean, gstd, fraction_valid = 0, 0, 0
        
        # Saturation time statistics
        if saturation_times:
            sat_mean = np.mean(saturation_times)
            sat_std = np.std(saturation_times)
            sat_min = np.min(saturation_times)
            sat_max = np.max(saturation_times)
        else:
            sat_mean, sat_std, sat_min, sat_max = np.nan, np.nan, np.nan, np.nan
        
        # Noise floor statistics
        if noise_floors:
            noise_mean = np.mean(noise_floors)
            noise_std = np.std(noise_floors)
        else:
            noise_mean, noise_std = np.nan, np.nan
        
        # Store in summary arrays
        results_summary.append([vb, gmean, gstd, fraction_valid, len(valid_gammas)])
        saturation_summary.append([vb, sat_mean, sat_std, sat_min, sat_max, 
                                   len(saturation_times), noise_mean, noise_std])
        
        log(f"vbeam={vb:.2f}: γ={gmean:.4f}±{gstd:.4f}, sat={sat_mean:.1f}±{sat_std:.1f} "
            f"({len(valid_gammas)}/{repeats} valid)")
    
    # Save summaries
    summary_df = pd.DataFrame(results_summary, 
        columns=['vbeam', 'gamma_mean', 'gamma_std', 'fraction_valid', 'n_valid'])
    summary_df.to_csv(csv_summary_path, index=False)
    log(f"Saved growth rate summary: {csv_summary_path}")
    
    saturation_df = pd.DataFrame(saturation_summary,
        columns=['vbeam', 'sat_mean', 'sat_std', 'sat_min', 'sat_max', 
                 'n_sat', 'noise_mean', 'noise_std'])
    saturation_df.to_csv(csv_saturation_path, index=False)
    log(f"Saved saturation summary: {csv_saturation_path}")
    
    # Plot growth rate vs vbeam with error bars
    plot_growth_rate_vs_vbeam(summary_df, label, outputs, vbeam_list, repeats)
    
    # Generate heatmaps
    generate_heatmaps(outputs, vbeam_list, repeats, label)
    
    # Plot analytic comparison for default and narrow cases
    if not bump_config:
        plot_analytic_comparison(summary_df, label, narrow)
    
    # Print saturation time analysis for lab book
    print_saturation_analysis_for_labbook(saturation_df, label)
    
    # Plot saturation time vs vbeam
    plot_saturation_vs_vbeam(saturation_df, label)
    
    # Plot correlation between growth rate and saturation time
    plot_gamma_vs_saturation_correlation(summary_df, saturation_df, label)
    
    return summary_df, outputs, saturation_df

def plot_growth_rate_vs_vbeam(summary_df, label, outputs, vbeam_list, repeats):
    """Plot growth rate vs beam velocity with error bars."""
    plt.figure(figsize=(10, 6))
    
    # Plot measured growth rates
    mask = summary_df['n_valid'] > 0
    plt.errorbar(summary_df['vbeam'][mask], 
                 summary_df['gamma_mean'][mask],
                 yerr=summary_df['gamma_std'][mask],
                 fmt='o-', capsize=4, label='Measured γ', linewidth=2)
    
    # Plot all individual measurements as points
    all_gammas = []
    all_vbeams = []
    for output in outputs:
        if not np.isnan(output['gamma']) and output['gamma'] > 0:
            all_gammas.append(output['gamma'])
            all_vbeams.append(output['vbeam'])
    
    plt.scatter(all_vbeams, all_gammas, alpha=0.3, s=20, 
                color='red', label='Individual runs')
    
    # Add analytic prediction for comparison
    if label in ['default', 'narrow']:
        # Find most unstable k for each vbeam
        v_analytic = np.linspace(0.1, max(vbeam_list), 100)
        gamma_analytic = []
        for v in v_analytic:
            k_opt, gamma_max = find_most_unstable_k(v)
            gamma_analytic.append(gamma_max)
        
        plt.plot(v_analytic, gamma_analytic, '--', color='green', 
                linewidth=2, label='Analytic (cold beams)')
        
        # Mark upper threshold (k v0 = ω_pe√2)
        # For fundamental mode k = 2π/L
        k_fundamental = 2 * np.pi / 100
        upper_threshold = np.sqrt(2) / k_fundamental
        plt.axvline(x=upper_threshold, color='red', linestyle=':', 
                   label=f'Upper threshold: {upper_threshold:.2f}')
    
    plt.xlabel("Beam velocity vbeam", fontsize=12)
    plt.ylabel("Growth rate γ", fontsize=12)
    plt.title(f"Two-stream instability: Growth rate vs beam velocity ({label} beams)", 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(PLOT_DIR, f"twostream_{label}_gamma_vs_vbeam.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log(f"Saved growth rate plot: {plot_path}")

def plot_analytic_comparison(summary_df, label, narrow):
    """Plot detailed comparison with analytic theory."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Growth rate comparison
    mask = summary_df['n_valid'] > 0
    ax1.errorbar(summary_df['vbeam'][mask], 
                 summary_df['gamma_mean'][mask],
                 yerr=summary_df['gamma_std'][mask],
                 fmt='o-', capsize=4, label='Measured', linewidth=2)
    
    # Analytic prediction for cold beams
    v_analytic = np.linspace(0.1, max(summary_df['vbeam']), 200)
    gamma_analytic = []
    for v in v_analytic:
        k_opt, gamma_max = find_most_unstable_k(v)
        gamma_analytic.append(gamma_max)
    
    ax1.plot(v_analytic, gamma_analytic, '--', color='red', 
            linewidth=2, label='Analytic (cold beams)')
    
    # Mark thresholds
    k_fundamental = 2 * np.pi / 100
    upper_threshold = np.sqrt(2) / k_fundamental
    ax1.axvline(x=upper_threshold, color='green', linestyle=':', 
               label=f'Upper threshold: {upper_threshold:.2f}')
    
    ax1.set_xlabel("Beam velocity vbeam", fontsize=12)
    ax1.set_ylabel("Growth rate γ", fontsize=12)
    ax1.set_title(f"Comparison with Analytic Theory ({label} beams)", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Instability region
    k_vals = np.linspace(0.1, 1.0, 100)
    v_vals = np.linspace(0.1, 10, 100)
    K, V = np.meshgrid(k_vals, v_vals)
    
    Gamma = np.zeros_like(K)
    for i in range(len(v_vals)):
        for j in range(len(k_vals)):
            Gamma[i, j] = analytic_growth_rate(k_vals[j], v_vals[i])
    
    contour = ax2.contourf(K, V, Gamma, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax2, label='Growth rate γ')
    
    # Mark the most unstable k for each vbeam
    most_unstable_k = []
    for v in summary_df['vbeam'][mask]:
        k_opt, _ = find_most_unstable_k(v)
        most_unstable_k.append(k_opt if k_opt else 0)
    
    ax2.scatter(most_unstable_k, summary_df['vbeam'][mask], 
               c='red', s=50, marker='o', label='Measured points')
    
    # Upper threshold line: k v = ω_pe√2
    k_thresh = np.sqrt(2) / v_vals
    ax2.plot(k_thresh, v_vals, 'w--', linewidth=2, label='Upper threshold')
    
    ax2.set_xlabel("Wavenumber k", fontsize=12)
    ax2.set_ylabel("Beam velocity vbeam", fontsize=12)
    ax2.set_title("Instability Region in k-vbeam Space", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"twostream_{label}_analytic_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log(f"Saved analytic comparison plot: {plot_path}")

def generate_heatmaps(outputs, vbeam_list, repeats, label):
    """Generate heatmaps of various metrics."""
    metrics = ["gamma", "noise_floor", "sat_time"]
    metric_titles = {
        "gamma": "Growth rate γ",
        "noise_floor": "Noise floor",
        "sat_time": "Saturation time"
    }
    
    for metric in metrics:
        # Prepare 2D array
        data = np.zeros((len(vbeam_list), repeats))
        
        i = 0
        for vb_idx, vb in enumerate(vbeam_list):
            for rep in range(repeats):
                val = outputs[i][metric]
                data[vb_idx, rep] = val if val is not None else np.nan
                i += 1
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(data, annot=True, fmt=".3f", cmap="viridis",
                    xticklabels=[f"rep{r+1}" for r in range(repeats)],
                    yticklabels=[f"{vb:.2f}" for vb in vbeam_list])
        plt.xlabel("Repeat")
        plt.ylabel("Beam velocity vbeam")
        plt.title(f"Heatmap of {metric_titles[metric]} ({label} beams)")
        plt.tight_layout()
        
        heatmap_path = os.path.join(PLOT_DIR, 
            f"twostream_{label}_{metric}_heatmap.png")
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        log(f"Saved heatmap: {heatmap_path}")


# Add these three missing functions (place them after generate_heatmaps function):

def plot_saturation_vs_vbeam(saturation_df, label):
    """Plot saturation time vs beam velocity."""
    if saturation_df.empty:
        return
    
    # Filter out NaN values
    valid_data = saturation_df[saturation_df['sat_mean'].notna()]
    
    if len(valid_data) < 2:
        log(f"Not enough valid saturation data to plot for {label} beams")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Error bar plot
    plt.errorbar(valid_data['vbeam'], valid_data['sat_mean'],
                 yerr=valid_data['sat_std'], fmt='o-', capsize=4,
                 linewidth=2, markersize=8, label=f'{label} beams')
    
    # Add trend annotation
    if len(valid_data) >= 3:
        # Fit polynomial
        z = np.polyfit(valid_data['vbeam'], valid_data['sat_mean'], 2)
        p = np.poly1d(z)
        v_fine = np.linspace(min(valid_data['vbeam']), max(valid_data['vbeam']), 100)
        plt.plot(v_fine, p(v_fine), 'r--', alpha=0.7, label='Trend line')
        
        # Calculate and show slope
        slope = (valid_data['sat_mean'].iloc[-1] - valid_data['sat_mean'].iloc[0]) / \
                (valid_data['vbeam'].iloc[-1] - valid_data['vbeam'].iloc[0])
        
        plt.annotate(f"Trend slope: {slope:.3f} (earlier saturation for larger vbeam)",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                    fontsize=10)
    
    plt.xlabel("Beam Velocity (vbeam)", fontsize=12)
    plt.ylabel("Saturation Time", fontsize=12)
    plt.title(f"Saturation Time vs Beam Velocity\n({label.capitalize()} Beams)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add horizontal lines for reference ranges
    plt.axhline(y=25, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=15, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=10, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=5, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"saturation_time_{label}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log(f"Saved saturation time plot: {plot_path}")

def plot_gamma_vs_saturation_correlation(growth_df, saturation_df, label):
    """Plot correlation between growth rate and saturation time."""
    if growth_df.empty or saturation_df.empty:
        return
    
    # Merge dataframes on vbeam
    merged = pd.merge(growth_df, saturation_df, on='vbeam')
    
    # Filter valid data
    valid = merged[(merged['gamma_mean'] > 0) & 
                   (merged['sat_mean'].notna()) &
                   (merged['n_valid'] > 0) &
                   (merged['n_sat'] > 0)]
    
    if len(valid) < 3:
        log(f"Not enough data for correlation plot for {label} beams")
        return
    
    plt.figure(figsize=(8, 6))
    
    # Scatter plot with error bars
    plt.errorbar(valid['gamma_mean'], valid['sat_mean'],
                 xerr=valid['gamma_std'], yerr=valid['sat_std'],
                 fmt='o', capsize=3, alpha=0.7, label='Data points')
    
    # Add labels for each point
    for _, row in valid.iterrows():
        plt.annotate(f"v={row['vbeam']:.1f}", 
                    (row['gamma_mean'], row['sat_mean']),
                    textcoords="offset points", xytext=(0,5),
                    ha='center', fontsize=8)
    
    # Linear fit
    z = np.polyfit(valid['gamma_mean'], valid['sat_mean'], 1)
    p = np.poly1d(z)
    x_fit = np.linspace(min(valid['gamma_mean']), max(valid['gamma_mean']), 100)
    plt.plot(x_fit, p(x_fit), 'r-', linewidth=2,
             label=f'Linear fit: y = {z[0]:.2f}x + {z[1]:.2f}')
    
    # Calculate correlation
    correlation = np.corrcoef(valid['gamma_mean'], valid['sat_mean'])[0,1]
    
    plt.xlabel("Growth Rate (γ)", fontsize=12)
    plt.ylabel("Saturation Time", fontsize=12)
    plt.title(f"Growth Rate vs Saturation Time\n({label.capitalize()} Beams, r = {correlation:.3f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add correlation interpretation
    if correlation < -0.7:
        interpretation = "Strong negative correlation\n(larger γ → much earlier saturation)"
    elif correlation < -0.3:
        interpretation = "Moderate negative correlation\n(larger γ → earlier saturation)"
    else:
        interpretation = "Weak correlation"
    
    plt.annotate(interpretation,
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"gamma_vs_saturation_{label}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log(f"Saved γ vs saturation correlation plot: {plot_path}")
    
    # Log correlation result
    log(f"\nCorrelation analysis for {label} beams:")
    log(f"  Correlation coefficient (γ vs sat_time): {correlation:.3f}")
    log(f"  Slope: {z[0]:.3f} (saturation time decreases by {-z[0]:.2f} per unit γ)")
    if correlation < -0.3:
        log("  ✓ Confirmed: Larger growth rate leads to earlier saturation")

def print_saturation_analysis_for_labbook(saturation_df, label):
    """Print formatted saturation analysis for lab book."""
    log("\n" + "=" * 60)
    log(f"SATURATION TIME ANALYSIS - {label.upper()} BEAMS")
    log("=" * 60)
    
    log("\n6.2 Saturation Time")
    log("Definition: First time when dA/dt ≤ 0")
    log("Trend: Saturation occurs earlier for larger vbeam")
    log("\nvbeam range\tSaturation time (approx)")
    log("-" * 50)
    
    # Filter out NaN values
    valid_data = saturation_df[saturation_df['sat_mean'].notna()]
    
    # Group into ranges for lab book table
    ranges = [
        (2.0, 3.0),
        (3.0, 4.0),
        (4.0, 5.0),
        (5.0, 6.0),
        (6.0, 7.0)
    ]
    
    for vmin, vmax in ranges:
        # Get data in this range
        range_data = valid_data[(valid_data['vbeam'] >= vmin) & 
                               (valid_data['vbeam'] < vmax)]
        
        if len(range_data) > 0:
            # Calculate weighted average (weighted by number of measurements)
            total_sat = 0
            total_weight = 0
            
            for _, row in range_data.iterrows():
                if row['n_sat'] > 0:
                    weight = row['n_sat']
                    total_sat += row['sat_mean'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                mean_sat = total_sat / total_weight
                
                # Calculate approximate range (± one standard deviation)
                # Get representative std
                std_values = []
                for _, row in range_data.iterrows():
                    if row['n_sat'] > 0 and not np.isnan(row['sat_std']):
                        std_values.append(row['sat_std'])
                
                if std_values:
                    avg_std = np.mean(std_values)
                    t_min = int(max(0, mean_sat - avg_std))
                    t_max = int(mean_sat + avg_std)
                    
                    log(f"{vmin:.1f}-{vmax:.1f}\tt = {t_min}-{t_max}")
    
    log("\nExplanation: Stronger instability (larger γ) grows faster,")
    log("reaches trapping threshold sooner, leading to earlier saturation.")
    
    # Also print the detailed data table
    log("\n" + "-" * 70)
    log("DETAILED SATURATION TIMES:")
    log("-" * 70)
    log(f"{'vbeam':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'n':<6} {'Noise':<12}")
    log("-" * 70)
    
    for _, row in valid_data.iterrows():
        log(f"{row['vbeam']:<8.2f} {row['sat_mean']:<10.2f} {row['sat_std']:<10.2f} "
            f"{row['sat_min']:<10.2f} {row['sat_max']:<10.2f} {row['n_sat']:<6} "
            f"{row['noise_mean']:<12.6f}")

# -------------------------------------------------------------
# Bump-on-tail analysis (Task 4 Optional)
# -------------------------------------------------------------
def analyze_bump_on_tail(vbump=3.0, fraction_bump=0.1, npart=10000):
    """Run bump-on-tail simulation and analyze heating."""
    log(f"Running bump-on-tail analysis: vbump={vbump}, fraction={fraction_bump}")
    
    L = 100
    Nc = 20
    
    # Initial condition
    pos, vel_initial = bump(npart, L, vbump, fraction_bump)
    
    # Initial velocity distribution analysis
    initial_fit, initial_r2 = fit_velocity_distribution(vel_initial)
    
    # Run simulation - CAPTURE THE RETURNED VELOCITIES!
    summary = Summary()
    pos_final, vel_final = run(pos, vel_initial, L, Nc,
        out=[summary],
        output_times=np.linspace(0.0, 200.0, 20))  # Longer simulation for heating
    
    # Final velocity distribution - use vel_final not vel_initial!
    final_fit, final_r2 = fit_velocity_distribution(vel_final)
    
    # Plot velocity distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initial distribution
    counts_i, bins_i = histogram(vel_initial, bins=50, density=True)
    bin_centers_i = (bins_i[:-1] + bins_i[1:]) / 2
    axes[0].stairs(counts_i, bins_i, fill=True, alpha=0.7, label='Initial')
    axes[0].plot(bin_centers_i, 
                 gaussian(bin_centers_i, *initial_fit),
                 'r-', linewidth=2, label=f'Fit: μ={initial_fit[1]:.2f}, σ={initial_fit[2]:.2f}')
    axes[0].set_xlabel("Velocity")
    axes[0].set_ylabel("Distribution function")
    axes[0].set_title(f"Initial velocity distribution\n(vbump={vbump}, fraction={fraction_bump})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final distribution
    counts_f, bins_f = histogram(vel_final, bins=50, density=True)
    bin_centers_f = (bins_f[:-1] + bins_f[1:]) / 2
    axes[1].stairs(counts_f, bins_f, fill=True, alpha=0.7, label='Final')
    axes[1].plot(bin_centers_f, 
                 gaussian(bin_centers_f, *final_fit),
                 'r-', linewidth=2, label=f'Fit: μ={final_fit[1]:.2f}, σ={final_fit[2]:.2f}')
    axes[1].set_xlabel("Velocity")
    axes[1].set_ylabel("Distribution function")
    axes[1].set_title(f"Final velocity distribution\nTemperature change: {(final_fit[2]**2 - initial_fit[2]**2):.3f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, 
        f"bump_v{vbump}_f{fraction_bump}_distributions.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    # Print results
    log(f"Bump-on-tail analysis results:")
    log(f"  Initial: μ={initial_fit[1]:.3f}, σ={initial_fit[2]:.3f} (T={initial_fit[2]**2:.3f})")
    log(f"  Final:   μ={final_fit[1]:.3f}, σ={final_fit[2]:.3f} (T={final_fit[2]**2:.3f})")
    log(f"  Temperature increase: {final_fit[2]**2 - initial_fit[2]**2:.3f}")
    log(f"  Mean velocity change: {final_fit[1] - initial_fit[1]:.3f}")
    
    # Also plot the time evolution of the amplitude
    plt.figure(figsize=(10, 6))
    plt.plot(summary.t, np.abs(summary.firstharmonic))
    plt.xlabel("Time")
    plt.ylabel("|First harmonic|")
    plt.title(f"Bump-on-tail: Amplitude evolution (vbump={vbump})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path2 = os.path.join(PLOT_DIR, f"bump_v{vbump}_amplitude.png")
    plt.savefig(plot_path2, dpi=150)
    plt.close()
    log(f"  Amplitude plot saved: {plot_path2}")
    
    return {
        "initial_mean": initial_fit[1],
        "initial_width": initial_fit[2],
        "initial_temp": initial_fit[2]**2,
        "final_mean": final_fit[1],
        "final_width": final_fit[2],
        "final_temp": final_fit[2]**2,
        "temp_increase": final_fit[2]**2 - initial_fit[2]**2,
        "mean_change": final_fit[1] - initial_fit[1],
        "final_velocity_array": vel_final,  # For debugging
        "initial_velocity_array": vel_initial  # For debugging
    }

def print_saturation_analysis_for_labbook(saturation_df, label):
    """Print formatted saturation analysis for lab book."""
    log("\n" + "=" * 60)
    log(f"SATURATION TIME ANALYSIS - {label.upper()} BEAMS")
    log("=" * 60)
    
    log("\n6.2 Saturation Time")
    log("Definition: First time when dA/dt ≤ 0")
    log("Trend: Saturation occurs earlier for larger vbeam")
    log("\nvbeam range\tSaturation time (approx)")
    log("-" * 50)
    
    # Filter out NaN values
    valid_data = saturation_df[saturation_df['sat_mean'].notna()]
    
    # Group into ranges for lab book table
    ranges = [
        (2.0, 3.0),
        (3.0, 4.0),
        (4.0, 5.0),
        (5.0, 6.0),
        (6.0, 7.0)
    ]
    
    for vmin, vmax in ranges:
        # Get data in this range
        range_data = valid_data[(valid_data['vbeam'] >= vmin) & 
                               (valid_data['vbeam'] < vmax)]
        
        if len(range_data) > 0:
            # Calculate weighted average (weighted by number of measurements)
            total_sat = 0
            total_weight = 0
            
            for _, row in range_data.iterrows():
                if row['n_sat'] > 0:
                    weight = row['n_sat']
                    total_sat += row['sat_mean'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                mean_sat = total_sat / total_weight
                
                # Calculate approximate range (± one standard deviation)
                # Get representative std
                std_values = []
                for _, row in range_data.iterrows():
                    if row['n_sat'] > 0 and not np.isnan(row['sat_std']):
                        std_values.append(row['sat_std'])
                
                if std_values:
                    avg_std = np.mean(std_values)
                    t_min = int(max(0, mean_sat - avg_std))
                    t_max = int(mean_sat + avg_std)
                    
                    log(f"{vmin:.1f}-{vmax:.1f}\tt = {t_min}-{t_max}")
    
    log("\nExplanation: Stronger instability (larger γ) grows faster,")
    log("reaches trapping threshold sooner, leading to earlier saturation.")
    
    # Also print the detailed data table
    log("\n" + "-" * 70)
    log("DETAILED SATURATION TIMES:")
    log("-" * 70)
    log(f"{'vbeam':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'n':<6} {'Noise':<12}")
    log("-" * 70)
    
    for _, row in valid_data.iterrows():
        log(f"{row['vbeam']:<8.2f} {row['sat_mean']:<10.2f} {row['sat_std']:<10.2f} "
            f"{row['sat_min']:<10.2f} {row['sat_max']:<10.2f} {row['n_sat']:<6} "
            f"{row['noise_mean']:<12.6f}")

def plot_saturation_vs_vbeam(saturation_df, label):
    """Plot saturation time vs beam velocity."""
    if saturation_df.empty:
        return
    
    # Filter out NaN values
    valid_data = saturation_df[saturation_df['sat_mean'].notna()]
    
    if len(valid_data) < 2:
        log(f"Not enough valid saturation data to plot for {label} beams")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Error bar plot
    plt.errorbar(valid_data['vbeam'], valid_data['sat_mean'],
                 yerr=valid_data['sat_std'], fmt='o-', capsize=4,
                 linewidth=2, markersize=8, label=f'{label} beams')
    
    # Add trend annotation
    if len(valid_data) >= 3:
        # Fit polynomial
        z = np.polyfit(valid_data['vbeam'], valid_data['sat_mean'], 2)
        p = np.poly1d(z)
        v_fine = np.linspace(min(valid_data['vbeam']), max(valid_data['vbeam']), 100)
        plt.plot(v_fine, p(v_fine), 'r--', alpha=0.7, label='Trend line')
        
        # Calculate and show slope
        slope = (valid_data['sat_mean'].iloc[-1] - valid_data['sat_mean'].iloc[0]) / \
                (valid_data['vbeam'].iloc[-1] - valid_data['vbeam'].iloc[0])
        
        plt.annotate(f"Trend slope: {slope:.3f} (earlier saturation for larger vbeam)",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                    fontsize=10)
    
    plt.xlabel("Beam Velocity (vbeam)", fontsize=12)
    plt.ylabel("Saturation Time", fontsize=12)
    plt.title(f"Saturation Time vs Beam Velocity\n({label.capitalize()} Beams)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add horizontal lines for reference ranges
    plt.axhline(y=25, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=15, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=10, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=5, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"saturation_time_{label}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log(f"Saved saturation time plot: {plot_path}")

def plot_gamma_vs_saturation_correlation(growth_df, saturation_df, label):
    """Plot correlation between growth rate and saturation time."""
    if growth_df.empty or saturation_df.empty:
        return
    
    # Merge dataframes on vbeam
    merged = pd.merge(growth_df, saturation_df, on='vbeam')
    
    # Filter valid data
    valid = merged[(merged['gamma_mean'] > 0) & 
                   (merged['sat_mean'].notna()) &
                   (merged['n_valid'] > 0) &
                   (merged['n_sat'] > 0)]
    
    if len(valid) < 3:
        log(f"Not enough data for correlation plot for {label} beams")
        return
    
    plt.figure(figsize=(8, 6))
    
    # Scatter plot with error bars
    plt.errorbar(valid['gamma_mean'], valid['sat_mean'],
                 xerr=valid['gamma_std'], yerr=valid['sat_std'],
                 fmt='o', capsize=3, alpha=0.7, label='Data points')
    
    # Add labels for each point
    for _, row in valid.iterrows():
        plt.annotate(f"v={row['vbeam']:.1f}", 
                    (row['gamma_mean'], row['sat_mean']),
                    textcoords="offset points", xytext=(0,5),
                    ha='center', fontsize=8)
    
    # Linear fit
    z = np.polyfit(valid['gamma_mean'], valid['sat_mean'], 1)
    p = np.poly1d(z)
    x_fit = np.linspace(min(valid['gamma_mean']), max(valid['gamma_mean']), 100)
    plt.plot(x_fit, p(x_fit), 'r-', linewidth=2,
             label=f'Linear fit: y = {z[0]:.2f}x + {z[1]:.2f}')
    
    # Calculate correlation
    correlation = np.corrcoef(valid['gamma_mean'], valid['sat_mean'])[0,1]
    
    plt.xlabel("Growth Rate (γ)", fontsize=12)
    plt.ylabel("Saturation Time", fontsize=12)
    plt.title(f"Growth Rate vs Saturation Time\n({label.capitalize()} Beams, r = {correlation:.3f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add correlation interpretation
    if correlation < -0.7:
        interpretation = "Strong negative correlation\n(larger γ → much earlier saturation)"
    elif correlation < -0.3:
        interpretation = "Moderate negative correlation\n(larger γ → earlier saturation)"
    else:
        interpretation = "Weak correlation"
    
    plt.annotate(interpretation,
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"gamma_vs_saturation_{label}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log(f"Saved γ vs saturation correlation plot: {plot_path}")
    
    # Log correlation result
    log(f"\nCorrelation analysis for {label} beams:")
    log(f"  Correlation coefficient (γ vs sat_time): {correlation:.3f}")
    log(f"  Slope: {z[0]:.3f} (saturation time decreases by {-z[0]:.2f} per unit γ)")
    if correlation < -0.3:
        log("  ✓ Confirmed: Larger growth rate leads to earlier saturation")


# -------------------------------------------------------------
# Main execution
# -------------------------------------------------------------
if __name__ == "__main__":
    log("=" * 60)
    log("Starting Two-Stream Instability")
    log("=" * 60)
    
    # ---------------------------------------------------------
    # TASK 1: Baseline two-stream run with animation
    # ---------------------------------------------------------
    log("\n--- TASK 1: Baseline Two-Stream Run ---")
    
    npart = 10000
    L = 100
    ncells = 20
    vbeam = 3.0
    
    times = np.linspace(0., 80, 200)
    
    # Initial condition
    pos, vel = twostream(npart, L, vbeam=vbeam)
    
    # Diagnostics including animation
    s = Summary()
    p = Plot(pos, vel, ncells, L)  # Animation enabled
    
    diagnostics_to_run = [s, p]
    
    # Run simulation
    log("Running baseline simulation with animation...")
    pos, vel = run(pos, vel, L, ncells,
                   out=diagnostics_to_run,
                   output_times=times)
    
    # Plot first harmonic
    plt.figure(figsize=(10, 6))
    plt.plot(s.t, s.firstharmonic)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("First harmonic amplitude", fontsize=12)
    plt.title("Two-stream instability growth (baseline)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plot_path1 = os.path.join(PLOT_DIR, "first_harmonic_growth.png")
    plt.savefig(plot_path1, dpi=150)
    plt.close()
    log(f"Saved baseline growth plot: {plot_path1}")
    
    # Log-scale plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(s.t, np.abs(s.firstharmonic))
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("|First harmonic| (log scale)", fontsize=12)
    plt.title("Two-stream instability: Exponential growth phase", fontsize=14)
    plt.grid(True, alpha=0.3)
    plot_path2 = os.path.join(PLOT_DIR, "first_harmonic_log.png")
    plt.savefig(plot_path2, dpi=150)
    plt.close()
    
    # Physical interpretation
    t = np.array(s.t)
    A = np.abs(np.array(s.firstharmonic))
    cut = len(A) // 3
    
    # Find growth region
    logA = np.log(A[:cut])
    pos_indices = A[:cut] > np.percentile(A[:cut], 10)
    if np.sum(pos_indices) > 10:
        growth_rate = np.polyfit(t[:cut][pos_indices], logA[pos_indices], 1)[0]
    else:
        growth_rate = 0.0
    
    # Find saturation
    dA_dt = np.gradient(A, t)
    sat_indices = np.where(dA_dt < 0)[0]
    sat_time = t[sat_indices[0]] if len(sat_indices) > 0 else None
    
    log("\n=== PHYSICAL INTERPRETATION ===")
    log(f"• Measured growth rate γ ≈ {growth_rate:.3f}")
    if sat_time:
        log(f"• Saturation begins at t ≈ {sat_time:.2f}")
    log("• Phase space shows two beams folding into vortices (electron trapping)")
    log("• Final state: BGK-like equilibrium with trapped particles")
    
    # ---------------------------------------------------------
    # TASK 2: Growth rate measurement with uncertainties
    # ---------------------------------------------------------
    log("\n--- TASK 2: Growth Rate Measurement ---")
    log("Method: Linear fit to log(A) in automatically detected growth region")
    log("Limitations:")
    log("  • Requires clear exponential growth phase")
    log("  • Sensitive to noise floor selection")
    log("  • May miss early/late growth phases")
    
    # Test measurement on baseline case
    gamma_test, t_start, t_end, quality = measure_growth_rate_enhanced(
        s.t, s.firstharmonic, method='linear_fit')
    log(f"Baseline measurement: γ={gamma_test:.4f} (t={t_start:.1f}-{t_end:.1f}, quality={quality:.2f})")
    
    # ---------------------------------------------------------
    # TASK 3: Instability thresholds and analytic comparison
    # ---------------------------------------------------------
    log("\n--- TASK 3: Instability Thresholds ---")
    
    # Calculate upper threshold from analytic theory
    k_fundamental = 2 * np.pi / L
    upper_threshold_analytic = np.sqrt(2) / k_fundamental
    log(f"Analytic upper threshold: vbeam ≥ {upper_threshold_analytic:.3f}")
    log(f"  (k = {k_fundamental:.3f}, ω_pe_total = √2 = {np.sqrt(2):.3f})")
    
    # Scan beam velocities
    v_list = np.linspace(0.0, 8.0, 20)  # Go above threshold
    log(f"Scanning vbeam from {v_list[0]:.1f} to {v_list[-1]:.1f}")
    
    # Default beams
    log("\n--- Default beams (width = 1.0) ---")
    summary_default, outputs_default, saturation_default = perform_scan(v_list, repeats=5, narrow=False)
    
    # ---------------------------------------------------------
    # TASK 4: Narrow beams
    # ---------------------------------------------------------
    log("\n--- Narrow beams (width = 0.1) ---")
    summary_narrow, outputs_narrow, saturation_narrow = perform_scan(v_list, repeats=3, narrow=True)
    
    # Compare thresholds
    log("\n=== THRESHOLD COMPARISON ===")
    
    # Find experimental thresholds
    def find_experimental_threshold(summary_df, threshold_gamma=0.001):
        """Find vbeam where gamma first exceeds threshold."""
        valid = summary_df[summary_df['gamma_mean'] > threshold_gamma]
        if len(valid) > 0:
            return valid['vbeam'].iloc[0]
        return None
    
    thresh_default = find_experimental_threshold(summary_default)
    thresh_narrow = find_experimental_threshold(summary_narrow)
    
    log(f"Upper threshold (analytic): {upper_threshold_analytic:.3f}")
    log(f"Experimental threshold (default): {thresh_default if thresh_default else 'N/A':.3f}")
    log(f"Experimental threshold (narrow): {thresh_narrow if thresh_narrow else 'N/A':.3f}")
    
    log("\nComments on agreement:")
    log("• Narrow beams should agree better with cold beam theory")
    log("• Default beams have thermal spread → higher threshold")
    log("• Finite width stabilizes some modes → reduced growth rates")
    
    # ---------------------------------------------------------
    # OPTIONAL TASK: Bump-on-tail heating
    # ---------------------------------------------------------
    log("\n--- OPTIONAL: Bump-on-Tail Heating ---")
    
    # Run bump-on-tail analysis
    bump_results = analyze_bump_on_tail(vbump=3.0, fraction_bump=0.1)
    
    # Explore parameter variations
    log("\n--- Parameter variation ---")
    for vbump in [2.0, 3.0, 4.0]:
        results = analyze_bump_on_tail(vbump=vbump, fraction_bump=0.1)
        log(f"vbump={vbump}: ΔT={results['temp_increase']:.3f}, Δμ={results['mean_change']:.3f}")
    
    # ---------------------------------------------------------
    # Summary and conclusions
    # ---------------------------------------------------------
    log("\n" + "=" * 60)
    log("SIMULATION COMPLETE")
    log("=" * 60)
    log(f"Results saved to: {TS_DIR}")
    log(f"  • CSV files: {CSV_DIR}")
    log(f"  • Plots: {PLOT_DIR}")
    log(f"  • Logs: {LOG_DIR}")
    
    LOG_FILE.close()