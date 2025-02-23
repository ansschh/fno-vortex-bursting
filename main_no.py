import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 1. Inspect the Data
# ===============================
data = torch.load("vorticity_evolution_data.pt", weights_only=False)
metadata = data["metadata"]
simulations = data["simulations"]

print("=== Metadata ===")
for key, value in metadata.items():
    print(f"{key}: {value}")

# For our purposes, we will focus on the profiles with valid numerical values.
valid_profiles = []
for profile_name, sim_data in simulations.items():
    snapshots = np.array(sim_data["snapshots"])
    if not (np.isnan(snapshots).any()):
        valid_profiles.append(profile_name)
    print(f"\nProfile: {profile_name}")
    print(f"  Number of snapshots: {snapshots.shape[0]}")
    print(f"  Snapshot shape: {snapshots.shape[1:]}")
    print(f"  Statistics: min={np.nanmin(snapshots):.6f}, max={np.nanmax(snapshots):.6f}, "
          f"mean={np.nanmean(snapshots):.6f}, std={np.nanstd(snapshots):.6f}")
    # Plot first snapshot
    plt.figure(figsize=(6,4))
    plt.imshow(snapshots[0], cmap='jet')
    plt.title(f"Profile '{profile_name}' - First Snapshot")
    plt.colorbar()
    plt.show()

print("\nValid profiles (without NaN):", valid_profiles)

# ===============================
# 2. Construct the Dataset
# ===============================
# We'll create pairs: (u(t), u(t+Δt)) for each valid profile.
inputs_list = []
targets_list = []
dt_snapshot = 0.5  # snapshot time interval, as seen in the time_steps

for profile_name in valid_profiles:
    sim_data = simulations[profile_name]
    snapshots = torch.tensor(sim_data["snapshots"], dtype=torch.float32)  # shape: (num_snapshots, Nr, Nz)
    num_snapshots = snapshots.shape[0]
    for i in range(num_snapshots - 1):
        # Each sample is a single-channel image (vorticity field)
        inputs_list.append(snapshots[i].unsqueeze(0))     # shape: (1, Nr, Nz)
        targets_list.append(snapshots[i+1].unsqueeze(0))    # shape: (1, Nr, Nz)

# Stack samples: shape (N_samples, 1, Nr, Nz)
inputs = torch.stack(inputs_list)
targets = torch.stack(targets_list)
print(f"\nDataset: {inputs.shape[0]} samples, each with shape {inputs.shape[1:]}.")

# (Optionally, you could also store the associated profile type as a label for conditioning)

# Split the dataset into training and testing (80/20 split)
total_samples = inputs.shape[0]
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size

class VorticityDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

full_dataset = VorticityDataset(inputs, targets)
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}.")

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ===============================
# 3. Define an Energy Computation Function
# ===============================
def compute_energy(field):
    """
    Compute a simple energy measure from a vorticity field.
    Here we define energy as the mean of the square of the vorticity.
    field: tensor of shape (B, 1, Nr, Nz)
    Returns: energy tensor of shape (B,)
    """
    return torch.mean(field**2, dim=[1,2,3])

# ===============================
# 4. Instantiate the FNO Model
# ===============================
# Import the base FNO model. (We assume it is available as neuralop.models.fno.FNO)
from neuralop.models import FNO  # Adjust import as needed

# Create the FNO model. The constructor here expects n_modes as a tuple.
model = FNO(
    n_modes=(12, 12),      # Use 12 Fourier modes along each spatial dimension
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    n_layers=4,
    projection_channels=64  # Optional parameter if supported
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

print("\nFNO Model:")
print(model)

# ===============================
# 5. Define the Loss Function and Optimizer
# ===============================
# Use a relative L2 loss (as defined in the toy example) with a small epsilon.
class L2Loss(object):
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, x, y):
        num_examples = x.size(0)
        diff_norms = torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), p=2, dim=1)
        y_norms = torch.norm(y.view(num_examples, -1), p=2, dim=1)
        return torch.sum(diff_norms / (y_norms + self.eps))

criterion = L2Loss(eps=1e-8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ===============================
# 6. Training Loop for the FNO Model
# ===============================
num_epochs = 50
train_losses = []
test_losses = []
best_test_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    for inputs_batch, targets_batch in train_loader:
        inputs_batch = inputs_batch.to(device)
        targets_batch = targets_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs_batch)
        loss = criterion(outputs, targets_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_train_loss += loss.item() * inputs_batch.size(0)
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    model.eval()
    epoch_test_loss = 0.0
    with torch.no_grad():
        for inputs_batch, targets_batch in test_loader:
            inputs_batch = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)
            outputs = model(inputs_batch)
            loss = criterion(outputs, targets_batch)
            epoch_test_loss += loss.item() * inputs_batch.size(0)
    epoch_test_loss /= len(test_loader.dataset)
    test_losses.append(epoch_test_loss)
    
    if epoch_test_loss < best_test_loss:
        best_test_loss = epoch_test_loss
        torch.save(model.state_dict(), 'best_fno_model.pt')
    
    print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss = {epoch_train_loss:.6f}, Test Loss = {epoch_test_loss:.6f}")

# ===============================
# 7. Evaluate and Visualize a Prediction & Energy Evolution
# ===============================
# Load best model and evaluate on test set
model.load_state_dict(torch.load("best_fno_model.pt"))
model.eval()

with torch.no_grad():
    # Get one batch from test_loader
    sample_input, sample_target = next(iter(test_loader))
    sample_input = sample_input.to(device)
    sample_target = sample_target.to(device)
    sample_pred = model(sample_input)

print("\nVisualizing sample prediction from test set:")
def visualize_snapshot(field, title=""):
    plt.figure(figsize=(6,4))
    plt.imshow(field.cpu().detach().squeeze(), cmap='jet')
    plt.title(title)
    plt.colorbar()
    plt.show()

visualize_snapshot(sample_input[0], title="Input Snapshot")
visualize_snapshot(sample_target[0], title="Target Snapshot")
visualize_snapshot(sample_pred[0], title="Predicted Snapshot")

# Compute and plot energy evolution on the test set:
energies_target = []
energies_pred = []
with torch.no_grad():
    for inputs_batch, targets_batch in test_loader:
        inputs_batch = inputs_batch.to(device)
        targets_batch = targets_batch.to(device)
        outputs = model(inputs_batch)
        energies_target.append(compute_energy(targets_batch))
        energies_pred.append(compute_energy(outputs))
energies_target = torch.cat(energies_target).cpu().numpy()
energies_pred = torch.cat(energies_pred).cpu().numpy()

plt.figure(figsize=(8,5))
plt.plot(energies_target, label="Target Energy", marker='o')
plt.plot(energies_pred, label="Predicted Energy", marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Energy (Mean Square Vorticity)")
plt.legend()
plt.title("Energy Evolution on Test Set")
plt.show()

# ===============================
# 8. Plot Loss History
# ===============================
plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Relative L2 Loss")
plt.legend()
plt.title("Training and Test Loss History")
plt.show()
