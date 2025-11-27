#!/usr/bin/env python3
#
# Electrostatic PIC code in a 1D cyclic domain
import numpy as np
import os
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
results_dir = os.path.abspath(results_dir)   # normalize
os.makedirs(results_dir, exist_ok=True)
from numpy import arange, concatenate, zeros, linspace, floor, array, pi
from numpy import sin, cos, sqrt, random, histogram, abs, sqrt, max

import matplotlib.pyplot as plt # Matplotlib plotting library
# For Peak detection
from scipy.signal import find_peaks

try:
    import matplotlib.gridspec as gridspec  # For plot layout grid
    got_gridspec = True
except:
    got_gridspec = False

# Need an FFT routine, either from SciPy or NumPy
try:
    from scipy.fftpack import fft, ifft
except:
    # No SciPy FFT routine. Import NumPy routine instead
    from numpy.fft import fft, ifft

def rk4step(f, y0, dt, args=()):
    """ Takes a single step using RK4 method """
    k1 = f(y0, *args)
    k2 = f(y0 + 0.5*dt*k1, *args)
    k3 = f(y0 + 0.5*dt*k2, *args)
    k4 = f(y0 + dt*k3, *args)

    return y0 + (k1 + 2.*k2 + 2.*k3 + k4)*dt / 6.

def calc_density(position, ncells, L):
    """ Calculate charge density given particle positions
    
    Input
      position  - Array of positions, one for each particle
                  assumed to be between 0 and L
      ncells    - Number of cells
      L         - Length of the domain

    Output
      density   - contains 1 if evenly distributed
    """
    # This is a crude method and could be made more efficient
    
    density = zeros([ncells])
    nparticles = len(position)
    
    dx = L / ncells       # Uniform cell spacing
    for p in position / dx:    # Loop over all the particles, converting position into a cell number
        plower = int(p)        # Cell to the left (rounding down)
        offset = p - plower    # Offset from the left
        density[plower] += 1. - offset
        density[(plower + 1) % ncells] += offset
    # nparticles now distributed amongst ncells
    density *= float(ncells) / float(nparticles)  # Make average density equal to 1
    return density

def periodic_interp(y, x):
    """
    Linear interpolation of a periodic array y at index x
    
    Input

    y - Array of values to be interpolated
    x - Index where result required. Can be an array of values
    
    Output
    
    y[x] with non-integer x
    """
    ny = len(y)
    if len(x) > 1:
        y = array(y) # Make sure it's a NumPy array for array indexing
    xl = floor(x).astype(int) # Left index
    dx = x - xl
    xl = ((xl % ny) + ny) % ny  # Ensures between 0 and ny-1 inclusive
    return y[xl]*(1. - dx) + y[(xl+1)%ny]*dx

def fft_integrate(y):
    """ Integrate a periodic function using FFTs
    """
    n = len(y) # Get the length of y
    
    f = fft(y) # Take FFT
    # Result is in standard layout with positive frequencies first then negative
    # n even: [ f(0), f(1), ... f(n/2), f(1-n/2) ... f(-1) ]
    # n odd:  [ f(0), f(1), ... f((n-1)/2), f(-(n-1)/2) ... f(-1) ]
    
    if n % 2 == 0: # If an even number of points
        k = concatenate( (arange(0, n/2+1), arange(1-n/2, 0)) )
    else:
        k = concatenate( (arange(0, (n-1)/2+1), arange( -(n-1)/2, 0)) )
    k = 2.*pi*k/n
    
    # Modify frequencies by dividing by ik
    f[1:] /= (1j * k[1:]) 
    f[0] = 0. # Set the arbitrary zero-frequency term to zero
    
    return ifft(f).real # Reverse Fourier Transform
   

def pic(f, ncells, L):
    """ f contains the position and velocity of all particles
    """
    nparticles = len(f) // 2     # Two values for each particle
    pos = f[0:nparticles] # Position of each particle
    vel = f[nparticles:]      # Velocity of each particle

    dx = L / float(ncells)    # Cell spacing

    # Ensure that pos is between 0 and L
    pos = ((pos % L) + L) % L
    
    # Calculate number density, normalised so 1 when uniform
    density = calc_density(pos, ncells, L)
    
    # Subtract ion density to get total charge density
    rho = density - 1.
    
    # Calculate electric field
    E = -fft_integrate(rho)*dx
    
    # Interpolate E field at particle locations
    accel = -periodic_interp(E, pos/dx)

    # Put back into a single array
    return concatenate( (vel, accel) )

####################################################################

def run(pos, vel, L, ncells=None, out=[], output_times=linspace(0,20,100), cfl=0.5):
    
    if ncells == None:
        ncells = int(sqrt(len(pos))) # A sensible default

    dx = L / float(ncells)
    
    f = concatenate( (pos, vel) )   # Starting state
    nparticles = len(pos)
    
    time = 0.0
    for tnext in output_times:
        # Advance to tnext
        stepping = True
        while stepping:
            # Maximum distance a particle can move is one cell
            dt = cfl * dx / max(abs(vel))
            if time + dt >= tnext:
                # Next time will hit or exceed required output time
                stepping = False
                dt = tnext - time
            f = rk4step(pic, f, dt, args=(ncells, L))
            time += dt
            
        # Extract position and velocities
        pos = ((f[0:nparticles] % L) + L) % L
        vel = f[nparticles:]
        
        # Send to output functions
        for func in out:
            func(pos, vel, ncells, L, time)
        
    return pos, vel

####################################################################
# 
# Output functions and classes
#

class Plot:
    """
    Displays three plots: phase space, charge density, and velocity distribution
    """
    def __init__(self, pos, vel, ncells, L):
        
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])
        
        # Plot initial positions
        if got_gridspec:
            self.fig = plt.figure()
            self.gs = gridspec.GridSpec(4, 4)
            ax = self.fig.add_subplot(self.gs[0:3,0:3])
            self.phase_plot = ax.plot(pos, vel, '.')[0]
            ax.set_title("Phase space")
            
            ax = self.fig.add_subplot(self.gs[3,0:3])
            self.density_plot = ax.plot(linspace(0, L, ncells), d)[0]
            
            ax = self.fig.add_subplot(self.gs[0:3,3])
            self.vel_plot = ax.plot(vhist, vbins)[0]
        else:
            self.fig = plt.figure()
            self.phase_plot = plt.plot(pos, vel, '.')[0]
            
            self.fig = plt.figure()
            self.density_plot = plt.plot(linspace(0, L, ncells), d)[0]
            
            self.fig = plt.figure()
            self.vel_plot = plt.plot(vhist, vbins)[0]
        plt.ion()
        plt.show()
        # Save the plot
        plot_animation_path = os.path.join(results_dir, "plot_animation.tif")
        plt.savefig(plot_animation_path, dpi=150)
        
    def __call__(self, pos, vel, ncells, L, t):
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])
        
        self.phase_plot.set_data(pos, vel) # Update the plot
        self.density_plot.set_data(linspace(0, L, ncells), d)
        self.vel_plot.set_data(vhist, vbins)
        plt.draw()
        plt.pause(0.05)
        

class Summary:
    def __init__(self):
        self.t = []
        self.firstharmonic = []
        
    def __call__(self, pos, vel, ncells, L, t):
        # Calculate the charge density
        d = calc_density(pos, ncells, L)
        
        # Amplitude of the first harmonic
        fh = 2.*abs(fft(d)[1]) / float(ncells)
        
        print(f"Time: {t} First: {fh}")
        
        self.t.append(t)
        self.firstharmonic.append(fh)

####################################################################
# 
# Functions to create the initial conditions
#

def landau(npart, L, alpha=0.2):
    """
    Creates the initial conditions for Landau damping
    
    """
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    pos0 = pos.copy()
    k = 2.*pi / L
    for i in range(10): # Adjust distribution using Newton iterations
        pos -= ( pos + alpha*sin(k*pos)/k - pos0 ) / ( 1. + alpha*cos(k*pos) )
        
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)
    
    return pos, vel

def twostream(npart, L, vbeam=2):
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)
    
    np2 = int(npart / 2)
    vel[:np2] += vbeam  # Half the particles moving one way
    vel[np2:] -= vbeam  # and half the other
    
    return pos,vel

####################################################################

if __name__ == "__main__":
    # Generate initial condition
    # 
    npart = 1000   
    if False:
        # 2-stream instability
        L = 100
        ncells = 20
        pos, vel = twostream(npart, L, 3.) # Might require more npart than Landau!
    else:
        # Landau damping
        L = 4.*pi
        ncells = 20
        pos, vel = landau(npart, L)
    
    # Create some output classes
    p = Plot(pos, vel, ncells, L) # This displays an animated figure - Slow!
    s = Summary()                 # Calculates, stores and prints summary info

    diagnostics_to_run = [s,p]   # Remove p to get much faster code!
    
    # Run the simulation
    pos, vel = run(pos, vel, L, ncells, 
                   out = diagnostics_to_run,        # These are called each output step
                   output_times=linspace(0.,20,50)) # The times to output
    # Save data
    np.savetxt(os.path.join(results_dir, "time.txt"), np.array(s.t))
    np.savetxt(os.path.join(results_dir, "harmonic.txt"), np.array(s.firstharmonic))
    # Detect the peak
    harmonic = np.array(s.firstharmonic)
    t = np.array(s.t)
    peaks, _ = find_peaks(harmonic)
    peak_times = t[peaks]
    peak_amps = harmonic[peaks]

    # Identify when noise dominates and split data
    noise_start_index = None
    for i in range(1, len(peak_amps)):
        if peak_amps[i] > peak_amps[i-1]:     # Noise now dominates
            noise_start_index = i
            break
    if noise_start_index is None:
        # No noise detected (rare case)
        signal = peak_amps.copy()
        noise  = np.array([])
        signal_times = peak_times.copy()
        noise_times  = np.array([])
    else:
        signal = peak_amps[:noise_start_index]
        noise  = peak_amps[noise_start_index:]
        signal_times = peak_times[:noise_start_index]
        noise_times  = peak_times[noise_start_index:]

    # ---------------------------------------------------------------
    # Compute and save noise level (Session 1 requirement)
    # ---------------------------------------------------------------
    if len(noise) > 1:
        noise_lvl = np.sqrt(np.mean(noise**2))
    else:
        noise_lvl = np.nan

    np.savetxt(
        os.path.join(results_dir, "noise_level.txt"),
        np.array([noise_lvl]),
        header="noise_rms"
    )

    print(f"Noise RMS = {noise_lvl}")


    # Save signal/noise data
    np.savetxt(os.path.join(results_dir, "signal_peaks.txt"),
            np.column_stack([signal_times, signal]),
            header="time amplitude")

    np.savetxt(os.path.join(results_dir, "noise_peaks.txt"),
            np.column_stack([noise_times, noise]),
            header="time amplitude")

    # Save peak data
    np.savetxt(os.path.join(results_dir, "peak_times.txt"), peak_times)
    np.savetxt(os.path.join(results_dir, "peak_amplitudes.txt"), peak_amps)

    print(f"Measured noise = {noise}")

    # Frequency measurement from signal peak spacing
    if len(signal_times) > 2:
        dt_peaks = np.diff(signal_times)

        mean_dt = np.mean(dt_peaks)
        std_dt  = np.std(dt_peaks, ddof=1)

        omega = 2 * np.pi / mean_dt
        omega_err = 2 * np.pi * std_dt / (mean_dt**2)

        np.savetxt(
            os.path.join(results_dir, "frequency.txt"),
            np.array([omega, omega_err]),
            header="omega omega_err"
        )

    print(f"Measured ω = {omega:.3f} ± {omega_err:.3f}")

    #   4. Fit exponential damping on signal region only
    # A(t) = A0 * exp(-d t)
    if len(signal) > 2:
        log_signal = np.log(signal)
        coeffs = np.polyfit(signal_times, log_signal, 1)
        slope, intercept = coeffs    # slope ≈ –damping_rate

        fitted_curve = np.exp(intercept + slope * signal_times)

        # Estimate uncertainty (standard error) of slope
        residuals = log_signal - (slope*signal_times + intercept)
        sigma2 = np.sum(residuals**2) / (len(signal_times) - 2)
        # cov = sigma2 * np.linalg.inv(np.dot(np.vstack([signal_times, np.ones(len(signal_times))]),
        # np.vstack([signal_times, np.ones(len(signal_times))]).T))
        X = np.vstack([signal_times, np.ones(len(signal_times))]).T
        cov = sigma2 * np.linalg.inv(X.T @ X)
        slope_error = np.sqrt(cov[0,0])
        gamma = -slope
        gamma_err = slope_error

        np.savetxt(os.path.join(results_dir, "damping_rate.txt"), np.array([gamma, gamma_err]))
        print(f"Measured y = {gamma:.3f} ± {gamma_err:.3f}")


    # Summary stores an array of the first-harmonic amplitude
    # Make a semilog plot to see exponential damping
    #1. Raw plot
    plt.figure()
    plt.plot(s.t, s.firstharmonic)
    plt.xlabel("Time [Normalised]")
    plt.ylabel("First harmonic amplitude [Normalised]")
    plt.yscale('log')
    # Save the plot
    plot_path = os.path.join(results_dir, "plot.png")
    plt.savefig(plot_path, dpi=150)    
    
    #2. Plot the peaks detection
    plt.figure(figsize=(8,5))
    plt.plot(t, harmonic, label="First Harmonic Amplitude")
    plt.scatter(peak_times, peak_amps, color='red', s=50, zorder=3, label="Detected Peaks")
    plt.yscale('log')
    plt.xlabel("Time [Normalised]")
    plt.ylabel("First Harmonic Amplitude [Normalised]")
    plt.title("Harmonic Amplitude with Peak Detection")
    plt.legend()

    # Save the plot with peaks detection
    peak_plot_path = os.path.join(results_dir, "peak_plot.png")
    plt.savefig(peak_plot_path, dpi=150)    
    

    # 3. Plot signal-only region vs noise-only region
    plt.figure(figsize=(8,5))
    plt.plot(peak_times, peak_amps, marker='x', label="Peaks (all)")
    plt.scatter(signal_times, signal, color="green", s=60, label="Signal Region")
    plt.scatter(noise_times,  noise,  color="red",   s=60, label="Noise Region")
    plt.yscale('log')
    plt.xlabel("Time")
    plt.ylabel("Peak Amplitude")
    plt.title("Signal vs Noise Peaks")
    plt.legend()
    plt.grid(True, alpha=0.3)

    signal_noise_plot = os.path.join(results_dir, "signal_vs_noise.png")
    plt.savefig(signal_noise_plot, dpi=150)

    #Plot exponential damping on signal region only
    plt.figure(figsize=(8,5))
    plt.scatter(signal_times, signal, label="Signal peaks", color='blue')
    plt.plot(signal_times, fitted_curve, '--', label=f"Fit: γ = {-slope:.3f}")
    plt.yscale('log')
    plt.xlabel("Time")
    plt.ylabel("Peak Amplitude")
    plt.title("Exponential Fit to Damping Region")
    plt.legend()
    plt.grid(True, alpha=0.3)

    fit_plot_path = os.path.join(results_dir, "signal_fit.png")
    plt.savefig(fit_plot_path, dpi=150)
    
    plt.ioff() # This so that the windows stay open

    plt.show()
    
    
