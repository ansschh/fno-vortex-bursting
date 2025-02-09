import numpy as np
import scipy.fftpack as fft
import time

# === Simulation Parameters (Same as Before) ===
N_r, N_z = 128, 256
L_r, L_z = 1.0, 4.0
dr = L_r / N_r
dz = L_z / N_z
dt = 0.01
num_steps = 500
nu = 1e-3

# === Helper Functions (Same as Before) ===
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

    k_squared = K_R**2 + K_Z**2
    k_squared[0, 0] = 1  # Avoid division by zero

    psi_hat = omega_hat / k_squared
    psi_hat[0, 0] = 0  # Ensure mean flow is zero

    u_r_hat = -1j * K_Z * psi_hat  # u_r = -∂ψ/∂z
    u_z_hat = 1j * K_R * psi_hat  # u_z = ∂ψ/∂r

    u_r = np.real(fft.ifft2(u_r_hat))
    u_z = np.real(fft.ifft2(u_z_hat))

    return u_r, u_z

def compute_convection(omega, u_r, u_z, dr, dz):
    """Computes the convection term (u · ∇ω) using finite differences."""
    d_omega_dr = (np.roll(omega, -1, axis=0) - np.roll(omega, 1, axis=0)) / (2 * dr)
    d_omega_dz = (np.roll(omega, -1, axis=1) - np.roll(omega, 1, axis=1)) / (2 * dz)
    return - (u_r * d_omega_dr + u_z * d_omega_dz)

# === Run a Single Numerical Simulation and Measure Time ===
core_size = 0.1  # Fixed value for consistency
perturbation_amplitude = 0.3
perturbation_wavelength = 1.0

omega = np.exp(-((np.linspace(0, L_r, N_r)[:, None] - 0.5) ** 2 + (np.linspace(0, L_z, N_z)[None, :] - 2.0) ** 2) / core_size ** 2)
omega *= (1 + perturbation_amplitude * np.sin(2 * np.pi * np.linspace(0, L_z, N_z)[None, :] / perturbation_wavelength))

start_time = time.time()

for t in range(num_steps):
    u_r, u_z = compute_velocity_fft(omega)
    convection_term = compute_convection(omega, u_r, u_z, dr, dz)
    diffusion_term = nu * laplacian(omega, dr, dz)

    # RK4 integration
    k1 = dt * (convection_term + diffusion_term)
    k2 = dt * (compute_convection(omega + 0.5 * k1, u_r, u_z, dr, dz) + nu * laplacian(omega + 0.5 * k1, dr, dz))
    k3 = dt * (compute_convection(omega + 0.5 * k2, u_r, u_z, dr, dz) + nu * laplacian(omega + 0.5 * k2, dr, dz))
    k4 = dt * (compute_convection(omega + k3, u_r, u_z, dr, dz) + nu * laplacian(omega + k3, dr, dz))
    
    omega += (k1 + 2 * k2 + 2 * k3 + k4) / 6  # RK4 update

end_time = time.time()
numerical_runtime = end_time - start_time

print(f"Numerical solver execution time for one simulation: {numerical_runtime:.4f} seconds")

# Save runtime for comparison
with open("numerical_runtime_single.txt", "w") as f:
    f.write(f"{numerical_runtime}\n")
