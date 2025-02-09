import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

# === Load Data in Batches to Avoid Memory Overload ===
def load_data_in_batches(file_path, batch_size):
    """Load large data in manageable batches."""
    data = np.load(file_path, mmap_mode='r')  # Use memory-mapped loading
    num_batches = data.shape[0] // batch_size
    return data, num_batches, batch_size

# === File Paths (Update if Necessary) ===
numerical_file = "vortex_data.npy"  # Numerical method results (Ground Truth)
fno_file = "fno_predictions.npy"  # FNO Predicted Results

if not os.path.exists(numerical_file) or not os.path.exists(fno_file):
    raise FileNotFoundError("Ensure both vortex_data.npy and fno_predictions.npy exist.")

# === Load numerical solution and FNO predictions ===
batch_size = 5  # Reduce batch size to prevent memory errors
numerical_data, num_batches, _ = load_data_in_batches(numerical_file, batch_size)
fno_predictions, _, _ = load_data_in_batches(fno_file, batch_size)

# === Get dataset shape ===
num_sims, timesteps, grid_r, grid_z = numerical_data.shape
assert numerical_data.shape == fno_predictions.shape, "Shape mismatch between FNO and ground truth!"

# === Compute MSE & SSIM in Batches ===
mse_per_timestep = np.zeros(timesteps)
ssim_per_timestep = np.zeros(timesteps)

for batch in range(num_batches):
    print(f"Processing Batch {batch+1}/{num_batches}...")
    
    true_batch = numerical_data[batch * batch_size:(batch + 1) * batch_size]
    pred_batch = fno_predictions[batch * batch_size:(batch + 1) * batch_size]
    
    for t in range(timesteps):
        true_field = true_batch[:, t]  # (batch_size, grid_r, grid_z)
        pred_field = pred_batch[:, t]  # (batch_size, grid_r, grid_z)
        
        # Compute Mean Squared Error (MSE)
        mse_per_timestep[t] += np.mean((true_field - pred_field) ** 2) / num_batches

        # Compute Structural Similarity Index (SSIM)
        ssim_values = [ssim(true_field[i], pred_field[i], data_range=true_field[i].max() - true_field[i].min()) for i in range(true_batch.shape[0])]
        ssim_per_timestep[t] += np.mean(ssim_values) / num_batches

# === Save Results to CSV ===
results_df = pd.DataFrame({
    "Time Step": range(timesteps),
    "MSE": mse_per_timestep,
    "SSIM": ssim_per_timestep
})
results_df.to_csv("fno_vs_numerical_results.csv", index=False)
print("Results saved to fno_vs_numerical_results.csv")

# === Plot MSE over Time ===
plt.figure(figsize=(8, 4))
plt.plot(range(timesteps), mse_per_timestep, marker='o', label="MSE", color="blue")
plt.xlabel("Time Step")
plt.ylabel("Mean Squared Error")
plt.title("MSE Between FNO and Numerical Method")
plt.legend()
plt.grid()
plt.savefig("mse_plot.png")
plt.show()

# === Plot SSIM over Time ===
plt.figure(figsize=(8, 4))
plt.plot(range(timesteps), ssim_per_timestep, marker='o', color='green', label="SSIM")
plt.xlabel("Time Step")
plt.ylabel("SSIM Score")
plt.title("SSIM Between FNO and Numerical Method")
plt.legend()
plt.grid()
plt.savefig("ssim_plot.png")
plt.show()

print("Plots saved as mse_plot.png and ssim_plot.png")