import numpy as np
import torch
from numpy.fft import fft2, ifft2, fftfreq

# -----------------------------
# Domain and Grid Setup
# -----------------------------
Lr = 2.0      # Domain size in r-direction
Lz = 10.0     # Domain size in z-direction
Nr = 256      # Number of grid points in r
Nz = 512      # Number of grid points in z
dr = Lr / Nr
dz = Lz / Nz

# Create grid arrays: r (radial) and z (axial)
r = np.linspace(-Lr/2, Lr/2, Nr)
z = np.linspace(0, Lz, Nz)
R, Z = np.meshgrid(r, z, indexing='ij')

# -----------------------------
# Vortex Tube Parameters
# -----------------------------
Gamma0 = 0.5       # Circulation (reduced for slower advection)
sigma_min = 0.1    # Minimum (unperturbed) core size
A = 3.0            # Core size ratio: sigma_max/sigma_min
sigma_max = A * sigma_min
lambda_core = Lz / 2  # Wavelength for core size perturbation

# -----------------------------
# Simulation Parameters
# -----------------------------
nu = 1e-4    # Kinematic viscosity (reduced for slower evolution)
dt = 0.01    # Time step
T = 10.0     # Total simulation time
num_steps = int(T / dt)
snapshot_interval = 50  # Save a snapshot every 50 time steps

# -----------------------------
# FFT Setup for Poisson Solver
# -----------------------------
kx = 2 * np.pi * fftfreq(Nr, d=dr)
kz = 2 * np.pi * fftfreq(Nz, d=dz)
KX, KZ = np.meshgrid(kx, kz, indexing='ij')
K2 = KX**2 + KZ**2
K2[0, 0] = 1  # Avoid division by zero

# -----------------------------
# Profile Functions
# -----------------------------
def get_sigma(z_val, profile_type):
    """
    Return the vortex core size function sigma(z) based on the chosen profile type.
    
    profile_type can be one of:
      - "sinusoidal": sigma = sigma_min + 0.5*(sigma_max - sigma_min)*(1 + sin(...))
      - "cosine":     sigma = sigma_min + 0.5*(sigma_max - sigma_min)*(1 + cos(...))
      - "gaussian":   sigma = sigma_min + (sigma_max - sigma_min)*exp(-((z - Lz/2)^2)/(2*(lambda_core/4)^2))
      - "none":       sigma = sigma_min (unperturbed)
    """
    if profile_type == "sinusoidal":
        return sigma_min + 0.5 * (sigma_max - sigma_min) * (1 + np.sin(2 * np.pi * (z_val - Lz/2) / lambda_core))
    elif profile_type == "cosine":
        return sigma_min + 0.5 * (sigma_max - sigma_min) * (1 + np.cos(2 * np.pi * (z_val - Lz/2) / lambda_core))
    elif profile_type == "gaussian":
        std = lambda_core / 4
        return sigma_min + (sigma_max - sigma_min) * np.exp(-((z_val - Lz/2)**2) / (2 * std**2))
    elif profile_type == "none":
        return sigma_min * np.ones_like(z_val)
    else:
        raise ValueError("Unknown profile type: " + profile_type)

# -----------------------------
# Poisson Solver and Helpers
# -----------------------------
def solve_poisson(omega_field):
    omega_hat = fft2(omega_field)
    psi_hat = -omega_hat / K2
    return np.real(ifft2(psi_hat))

def compute_velocity(psi):
    psi_hat = fft2(psi)
    u_r_hat = -1j * KZ * psi_hat
    u_z_hat =  1j * KX * psi_hat
    u_r = np.real(ifft2(u_r_hat))
    u_z = np.real(ifft2(u_z_hat))
    return u_r, u_z

def laplacian(f):
    f_hat = fft2(f)
    return np.real(ifft2(-K2 * f_hat))

def rhs(omega_field, u_r, u_z, nu):
    dω_dr = np.gradient(omega_field, dr, axis=0)
    dω_dz = np.gradient(omega_field, dz, axis=1)
    advective = u_r * dω_dr + u_z * dω_dz
    diffusive = nu * laplacian(omega_field)
    return -advective + diffusive

def rk4_step(omega_field, dt, nu):
    psi = solve_poisson(omega_field)
    u_r, u_z = compute_velocity(psi)
    k1 = rhs(omega_field, u_r, u_z, nu)
    
    psi2 = solve_poisson(omega_field + 0.5 * dt * k1)
    u_r2, u_z2 = compute_velocity(psi2)
    k2 = rhs(omega_field + 0.5 * dt * k1, u_r2, u_z2, nu)
    
    psi3 = solve_poisson(omega_field + 0.5 * dt * k2)
    u_r3, u_z3 = compute_velocity(psi3)
    k3 = rhs(omega_field + 0.5 * dt * k2, u_r3, u_z3, nu)
    
    psi4 = solve_poisson(omega_field + dt * k3)
    u_r4, u_z4 = compute_velocity(psi4)
    k4 = rhs(omega_field + dt * k3, u_r4, u_z4, nu)
    
    return omega_field + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# -----------------------------
# Simulation Function
# -----------------------------
def run_simulation(profile_type):
    """
    Run the simulation for a given vortex core profile (profile_type) and return:
      - snapshots: evolution of the vorticity field (shape: [num_snapshots, Nr, Nz])
      - time_steps: time corresponding to each snapshot.
    """
    # Compute sigma for each z based on the chosen profile
    sigma_Z = get_sigma(Z, profile_type)
    # Initial condition: a Gaussian vorticity profile with a z-dependent core size.
    omega_field = (Gamma0 / (np.pi * sigma_Z**2)) * np.exp(- (R**2) / (sigma_Z**2))
    
    snapshots = []
    time_steps = []
    for step in range(num_steps):
        omega_field = rk4_step(omega_field, dt, nu)
        if step % snapshot_interval == 0:
            snapshots.append(omega_field.copy())
            time_steps.append(step * dt)
    snapshots = np.array(snapshots)  # Shape: (num_snapshots, Nr, Nz)
    time_steps = np.array(time_steps)
    return snapshots, time_steps

# -----------------------------
# Main: Run Simulations and Save Data
# -----------------------------
def main():
    profile_types = ["sinusoidal", "cosine", "gaussian", "none"]
    simulation_data = {}
    for profile in profile_types:
        print("Running simulation for profile:", profile)
        snapshots, time_steps = run_simulation(profile)
        simulation_data[profile] = {
            "time_steps": time_steps,
            "snapshots": snapshots  # Data shape: (num_snapshots, Nr, Nz)
        }
    
    # Metadata for reproducibility and future reference.
    metadata = {
        "Lr": Lr,
        "Lz": Lz,
        "Nr": Nr,
        "Nz": Nz,
        "dr": dr,
        "dz": dz,
        "Gamma0": Gamma0,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "lambda_core": lambda_core,
        "nu": nu,
        "dt": dt,
        "T": T,
        "num_steps": num_steps,
        "snapshot_interval": snapshot_interval,
        "profile_types": profile_types
    }
    
    data = {
        "metadata": metadata,
        "simulations": simulation_data
    }
    
    # Save the data to a PyTorch file (.pt)
    torch.save(data, "vorticity_evolution_data.pt")
    print("Simulation data saved to 'vorticity_evolution_data.pt'.")

if __name__ == '__main__':
    main()
