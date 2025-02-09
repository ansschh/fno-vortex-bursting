import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# === Load Data ===
data_file = "vortex_data.npy"  # Update filename if necessary
if not os.path.exists(data_file):
    raise FileNotFoundError(f"File {data_file} not found. Ensure your data is saved correctly.")

data = np.load(data_file)  # Shape: (num_simulations, timesteps, grid_r, grid_z)
print("Loaded Data Shape:", data.shape)  # Expected: (100, 10, 128, 256)

# Normalize Data
mean = np.mean(data)
std = np.std(data)
normalized_data = (data - mean) / std  # Normalize for stable training

# Convert to PyTorch Tensor
normalized_data = torch.tensor(normalized_data, dtype=torch.float32)


# === Define Dataset (Fixing 'Subset' Issue) ===
class VortexDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Shape: (num_simulations, timesteps, grid_r, grid_z)

    def __len__(self):
        return len(self.data)  # Corrected to work with Subset

    def __getitem__(self, idx):
        sample = self.data[idx]  # Extract one full time-series
        # Ensure shape: (timesteps, 1, height, width)
        return sample[:-1].permute(0, 2, 1).unsqueeze(1), sample[1:].permute(0, 2, 1).unsqueeze(1)  
        # Shape: (timesteps-1, 1, grid_r, grid_z) 


# === Create Full Dataset First ===
full_dataset = VortexDataset(normalized_data)

# === Split Data ===
train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# === Define FNO Model ===
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
        return self.fno(x)  # Ensure x is (batch, channels, height, width)


# Instantiate Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VortexFNO().to(device)

# === Training Setup ===
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Training Loop (Fixed Timesteps Processing) ===
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move to GPU if available

        optimizer.zero_grad()

        batch_size, timesteps, channels, height, width = inputs.shape
        outputs = torch.zeros_like(targets, device=device)  # Placeholder for outputs

        for t in range(timesteps):
            outputs[:, t] = model(inputs[:, t])  # Pass one timestep at a time

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss / len(train_loader):.6f}")

# === Evaluation Loop (Fixed Timesteps Processing) ===
model.eval()
test_loss = 0.0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        batch_size, timesteps, channels, height, width = inputs.shape
        outputs = torch.zeros_like(targets, device=device)

        for t in range(timesteps):
            outputs[:, t] = model(inputs[:, t])  # Process one timestep at a time

        loss = criterion(outputs, targets)
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_loader):.6f}")

# === Visualization (Fixed Shape Issues) ===
with torch.no_grad():
    sample_input, sample_target = next(iter(test_loader))
    sample_input, sample_target = sample_input.to(device), sample_target.to(device)
    
    batch_size, timesteps, channels, height, width = sample_input.shape
    predicted_output = torch.zeros_like(sample_target, device=device)

    for t in range(timesteps):
        predicted_output[:, t] = model(sample_input[:, t])  # Process timestep-wise

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(sample_input[0, 0, 0].cpu().numpy(), cmap='jet')  # Fix indexing
    ax[0].set_title("Input Vorticity Field")

    ax[1].imshow(sample_target[0, 0, 0].cpu().numpy(), cmap='jet')  # Fix indexing
    ax[1].set_title("True Evolution")

    ax[2].imshow(predicted_output[0, 0, 0].cpu().numpy(), cmap='jet')  # Fix indexing
    ax[2].set_title("Predicted Evolution")

    plt.show()

# === Save Model ===
torch.save(model.state_dict(), "vortex_fno.pth")
print("Model saved as vortex_fno.pth")

# === Load Model Later ===
model.load_state_dict(torch.load("vortex_fno.pth", map_location=device))
print("Model Loaded Successfully.")
