import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
import matplotlib as plt
import sys
matplotlib.use('Agg')

# --- GLOBAL CONFIGURATION (Needs to be defined inside the script) ---
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
NUM_FILES_TO_USE = 399990
save_dir = f"{NUM_FILES_TO_USE}files"

# --------------------------------------------------------------------

# LOADING DATASETS for cluster:
# FIX B: Use the correct cache filenames
try:
    train_data = np.load("train_data_files.npy")
    val_data = np.load("val_data_files.npy")
    print("Datasets loaded successfully!")
except FileNotFoundError:
    print("FATAL ERROR: Files not found. Please run the data processing step first.")
    # Exit gracefully if files aren't found
    sys.exit(1)

# Convert to Tensor
# Inputs: Cols 0 to 7 (8 features: m, L, J, n_v, d, B, n, Mass)
# Output: Col 8 (Radius)
X_train = torch.tensor(train_data[:, :8], dtype=torch.float32)
y_train = torch.tensor(train_data[:, 8:], dtype=torch.float32)

X_val = torch.tensor(val_data[:, :8], dtype=torch.float32)
y_val = torch.tensor(val_data[:, 8:], dtype=torch.float32)

# Separate Mass (index 7) from EOS parameters (indices 0-7)
X_eos_train, X_mass_train = X_train[:, :7], X_train[:, 7:]
X_eos_val, X_mass_val = X_val[:, :7], X_val[:, 7:]

# 1. Z-Score on EOS Parameters (m, L, J, n_v, d, B, n)
X_eos_mean = X_eos_train.mean(dim=0, keepdim=True)
X_eos_std = X_eos_train.std(dim=0, keepdim=True)
X_eos_std[X_eos_std == 0] = 1.0 

# Save them:
torch.save(X_eos_mean, f"X_eos_mean.pt")
torch.save(X_eos_std, f"X_eos_std.pt")
print("Normalization statistics saved.")

X_eos_train_norm = (X_eos_train - X_eos_mean) / X_eos_std
X_eos_val_norm = (X_eos_val - X_eos_mean) / X_eos_std

# 2. Constant Scaling on Mass (M)
MASS_SCALE = 3.5 
X_mass_train_norm = X_mass_train / MASS_SCALE
X_mass_val_norm = X_mass_val / MASS_SCALE

# 3. Recombine Inputs
X_train_norm = torch.cat((X_eos_train_norm, X_mass_train_norm), dim=1)
X_val_norm = torch.cat((X_eos_val_norm, X_mass_val_norm), dim=1)

# 1. Constant Scaling on Radius (R)
RADIUS_SCALE = 25.0

y_train_norm = y_train / RADIUS_SCALE
y_val_norm = y_val / RADIUS_SCALE

# Note: You no longer need y_mean and y_std for normalization, 
# but you MUST save RADIUS_SCALE to de-normalize predictions later.



class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim, auxiliary_dim=1):
        super().__init__()
        # We accept the hidden state + the auxiliary Mass injection
        self.fc = nn.Linear(hidden_dim + auxiliary_dim, hidden_dim)
        self.act = nn.GELU() 
    
    def forward(self, x, mass):
        # Concatenate Mass to the input of the layer
        combined = torch.cat([x, mass], dim=1)
        out = self.act(self.fc(combined))
        return x + out # Residual connection

class PhysicsEmulator(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=512): # Increased width
        super().__init__()
        # Separate EOS inputs from Mass
        # input_dim is 8: (7 EOS params + 1 Mass)
        self.eos_dim = input_dim - 1 
        
        # Initial encoding of EOS parameters only
        self.input_layer = nn.Linear(self.eos_dim, hidden_dim)
        
        # Deep Residual Layers with Mass Injection
        self.block1 = ResNetBlock(hidden_dim, auxiliary_dim=1)
        self.block2 = ResNetBlock(hidden_dim, auxiliary_dim=1)
        self.block3 = ResNetBlock(hidden_dim, auxiliary_dim=1)
        self.block4 = ResNetBlock(hidden_dim, auxiliary_dim=1)
        
        # Output layers
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2), # Inject mass one last time
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # Split input into EOS params and Mass
        # Assuming Mass is the LAST column (index -1)
        eos_params = x[:, :-1]
        mass = x[:, -1:]
        
        # 1. Encode EOS
        x_hidden = self.input_layer(eos_params)
        
        # 2. Pass through blocks, injecting Mass at each step
        x_hidden = self.block1(x_hidden, mass)
        x_hidden = self.block2(x_hidden, mass)
        x_hidden = self.block3(x_hidden, mass)
        x_hidden = self.block4(x_hidden, mass)
        
        # 3. Final Prediction
        # Concatenate mass one last time for the read-out
        combined_final = torch.cat([x_hidden, mass], dim=1)
        return self.final_layer(combined_final)



# --- Helper function for plotting (needs to be defined before calling) ---
def plot_and_save_losses(train_losses, val_losses, filename="loss_curve.png"):
    """Plots training and validation loss and saves the figure."""
    epochs = range(len(train_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss (Normalized)')
    plt.yscale('log') # Use log scale for clearer visualization of small losses
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    try:
        plt.savefig(filename)
        print(f"Loss plot saved to {filename}", flush=True)
    except Exception as e:
        print(f"ERROR saving plot: {e}", flush=True)
    plt.close() # Close the figure to free up memory
# --------------------------------------------------------------------------


model = PhysicsEmulator().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-7)
criterion = nn.HuberLoss()

# ==============================================================================
# 5. TRAINING
# ==============================================================================
train_loader = DataLoader(TensorDataset(X_train_norm, y_train_norm), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_norm, y_val_norm), batch_size=BATCH_SIZE, shuffle=False)

best_loss = float('inf')

# 1. ADD STORAGE LISTS
train_losses = []
val_losses = []
# Ensure RADIUS_SCALE is defined globally or passed in if running as a function
# If you didn't define save_dir, model will save in current directory.

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_b)
        loss = criterion(pred, y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            pred = model(X_b)
            loss = criterion(pred, y_b)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    scheduler.step()
    
    # 2. APPEND LOSSES
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        # Removed os.path.join(save_dir, "Best_EOS_Model.pth") for simplicity, assuming current directory
        torch.save(model.state_dict(), "Best_EOS_Model.pth")
    
    if epoch % 10 == 0:
        # Calculate the Approximate Physical Error in km
        phys_error_km = np.sqrt(2 * val_loss) * RADIUS_SCALE 
        
        print(f"Epoch {epoch} | Val Loss: {val_loss:.6e} | Approx Error: {phys_error_km:.4f} km", flush=True) 
        
        # 3. PLOT AND SAVE PERIODICALLY
        # Plot every 50 epochs (or choose a different interval)
        if epoch % 50 == 0 and epoch > 0:
            plot_and_save_losses(train_losses, val_losses, filename=f"loss_curve_epoch{epoch}.png")

# 4. FINAL PLOT after training finishes
plot_and_save_losses(train_losses, val_losses, filename="loss_curve_final.png")
print("Training complete. Best validation loss:", best_loss)