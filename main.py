import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# === Simulation Parameters ===
N_r, N_z = 128, 256  # Grid resolution (r, z)
L_r, L_z = 1.0, 4.0  # Physical domain size
dr = L_r / N_r
dz = L_z / N_z
dt = 0.01  # Time step
num_steps = 500  # Number of time steps
nu = 1e-3  # Viscosity
num_simulations = 100  # Number of different perturbation profiles

# === Create Spatial Grid ===
r = np.linspace(0, L_r, N_r)
z = np.linspace(0, L_z, N_z)
R, Z = np.meshgrid(r, z, indexing='ij')

# === Storage for Training Data ===
all_simulations = []

# === Helper Functions ===
def laplacian(omega, dr, dz):
    """Computes the 2D Laplacian using finite differences."""
    d2omega_dr2 = (np.roll(omega, -1, axis=0) - 2 * omega + np.roll(omega, 1, axis=0)) / dr**2
    d2omega_dz2 = (np.roll(omega, -1, axis=1) - 2 * omega + np.roll(omega, 1, axis=1)) / dz**2
    return d2omega_dr2 + d2omega_dz2

def compute_velocity_fft(omega):
    """Solves for the velocity field u using FFT-based Poisson solver (Biot-Savart Law)."""
    omega_hat = fft.fft2(omega)
    k_r = fft.fftfreq(N_r, d=dr) * 2 * np.pi
    k_z = fft.fftfreq(N_z, d=dz) * 2 * np.pi
    K_R, K_Z = np.meshgrid(k_r, k_z, indexing='ij')
    
    # Avoid division by zero at k=0
    k_squared = K_R**2 + K_Z**2
    k_squared[0, 0] = 1  # Set nonzero to prevent division error

    psi_hat = omega_hat / k_squared  # Stream function in Fourier space
    psi_hat[0, 0] = 0  # Ensure mean flow is zero
    
    # Compute velocity from stream function
    u_r_hat = -1j * K_Z * psi_hat  # u_r = -∂ψ/∂z
    u_z_hat =  1j * K_R * psi_hat  # u_z = ∂ψ/∂r

    u_r = np.real(fft.ifft2(u_r_hat))
    u_z = np.real(fft.ifft2(u_z_hat))
    
    return u_r, u_z

def compute_convection(omega, u_r, u_z, dr, dz):
    """Computes the convection term (u · ∇ω) using finite differences."""
    d_omega_dr = (np.roll(omega, -1, axis=0) - np.roll(omega, 1, axis=0)) / (2 * dr)
    d_omega_dz = (np.roll(omega, -1, axis=1) - np.roll(omega, 1, axis=1)) / (2 * dz)
    return - (u_r * d_omega_dr + u_z * d_omega_dz)

# === Run Multiple Simulations ===
for sim_id in range(num_simulations):
    print(f"Running simulation {sim_id + 1}/{num_simulations}...")

    # === Generate Different Initial Conditions ===
    core_size = np.random.uniform(0.05, 0.2)  # Random core size
    perturbation_amplitude = np.random.uniform(0.1, 0.5)  # Random amplitude
    perturbation_wavelength = np.random.uniform(0.5, 1.5)  # Random wavelength

    # Define initial vorticity profile (Gaussian core + sinusoidal perturbation)
    omega = np.exp(-((R - 0.5)**2 + (Z - 2.0)**2) / core_size**2)
    omega *= (1 + perturbation_amplitude * np.sin(2 * np.pi * Z / perturbation_wavelength))

    # Store time evolution
    omega_evolution = []

    # === Time Stepping Loop ===
    for t in tqdm(range(num_steps), desc=f"Sim {sim_id + 1} Progress"):
        # Compute velocity field from vorticity
        u_r, u_z = compute_velocity_fft(omega)

        # Compute convection term
        convection_term = compute_convection(omega, u_r, u_z, dr, dz)

        # Compute viscous term
        diffusion_term = nu * laplacian(omega, dr, dz)

        # Update vorticity using Runge-Kutta (RK4)
        k1 = dt * (convection_term + diffusion_term)
        k2 = dt * (compute_convection(omega + 0.5 * k1, u_r, u_z, dr, dz) + nu * laplacian(omega + 0.5 * k1, dr, dz))
        k3 = dt * (compute_convection(omega + 0.5 * k2, u_r, u_z, dr, dz) + nu * laplacian(omega + 0.5 * k2, dr, dz))
        k4 = dt * (compute_convection(omega + k3, u_r, u_z, dr, dz) + nu * laplacian(omega + k3, dr, dz))
        
        omega += (k1 + 2*k2 + 2*k3 + k4) / 6  # RK4 update

        # Store snapshot every 50 time steps
        if t % 50 == 0:
            omega_evolution.append(omega.copy())

    # Store full simulation results
    all_simulations.append(np.array(omega_evolution))

    # Save visualization for quick verification
    plt.figure(figsize=(8, 4))
    plt.imshow(omega, extent=[0, L_z, 0, L_r], origin="lower", cmap="coolwarm")
    plt.colorbar(label="Vorticity")
    plt.title(f"Final Vorticity Field - Sim {sim_id + 1}")
    plt.xlabel("z")
    plt.ylabel("r")
    plt.savefig(f"vortex_sim_{sim_id+1}.png")
    plt.close()

# === Save Data for Neural Operator Training ===
data_file = "vortex_data.npy"
np.save(data_file, np.array(all_simulations))
print(f"Data saved to {data_file}")
