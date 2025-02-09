import torch
import numpy as np
import time
import os
import torch.nn as nn

# === Define Model Structure (Same as Training) ===
class VortexFNO(nn.Module):
    def __init__(self, modes=12, width=32):
        super(VortexFNO, self).__init__()
        self.fno = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, 1, kernel_size=3, padding=1)  # Output channel: 1 (vorticity field)
        )

    def forward(self, x):
        return self.fno(x)

# === Load Model Properly ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VortexFNO().to(device)

# Load weights
model.load_state_dict(torch.load("vortex_fno.pth", map_location=device))
model.eval()  # Set model to evaluation mode
print("Model loaded successfully!")

# === Load Data ===
data_file = "vortex_data.npy"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"File {data_file} not found.")

data = np.load(data_file)  # Shape: (num_simulations, timesteps, grid_r, grid_z)
test_input = torch.tensor(data[0, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# === Measure Execution Time ===
start_time = time.time()
with torch.no_grad():
    prediction = model(test_input)
end_time = time.time()

fno_runtime = end_time - start_time
print(f"FNO execution time for one simulation: {fno_runtime:.4f} seconds")

# Save runtime
with open("fno_runtime_single.txt", "w") as f:
    f.write(f"{fno_runtime}\n")
