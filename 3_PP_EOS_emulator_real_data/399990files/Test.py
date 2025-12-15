import os
import re
import torch

import numpy as np

import torch.nn as nn
import glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# ==============================================================================
# 1. DEFINE THE MODEL
# ==============================================================================
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

# ==============================================================================
# 2. CONFIGURATION & HELPERS
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RADIUS_SCALE = 25.0
MASS_SCALE = 3.5
NUM_FILES_TO_USE = 399990
save_dir = f"{NUM_FILES_TO_USE}files"

# LOAD THE MODEL
model = PhysicsEmulator()
model.load_state_dict(torch.load("Best_EOS_Model.pth"))

model.eval()  # Set the model to evaluation mode
model.to(DEVICE)

# Regex must match the Training Pipeline exactly
pattern = re.compile(
    r"MREoSm(?P<m>\d+)"
    r"L(?P<L>\d+)"
    r"J(?P<J>\d+)"
    r"T(?P<T>\d+)_"
    r"n(?P<n>\d+)_"
    r"Yp?\+.*?RGgen_"
    r"v(?P<n_v>[-\d\.]+)"
    r"d(?P<d>[-\d\.]+)"
    r"B(?P<Btype>[np])(?P<B>\d+)\.dat"
)

def extract_eos_params(filename):
    match = pattern.match(filename)
    if not match:
        return None
    
    # Extract raw values exactly like training
    m = float(match.group("m")) / 100.0
    L = float(match.group("L"))
    J = float(match.group("J"))
    n_v = float(match.group("n_v"))
    d = float(match.group("d"))
    B = float(match.group("B")) / 1000.0
    n = float(match.group('n')) / 1000.0  # e.g., 0.160
    
    # Return as a numpy array for easy stacking [m, L, J, n_v, d, B, n]
    return np.array([m, L, J, n_v, d, B, n], dtype=np.float32)

# ==============================================================================
# 2. SELECT "UNSEEN" FILES
# ==============================================================================
# Path to your data
DATA_DIR_PATH = "/cluster/users/venkatek/ML/All_MR_Relations/All_MR_Relations/"
all_files = glob.glob(os.path.join(DATA_DIR_PATH, "MR*.dat"))
print("Total files found:", len(all_files))

# Load the list of files used for training
# Ensure this file exists from your training run!
try:
    training_files_used = np.load("files_used_for_training.npy", allow_pickle=True)
    # Convert to set for fast lookup
    training_set = set(training_files_used)
except FileNotFoundError:
    print("Warning: List of training files not found. Using random files (risk of data leakage).")
    training_set = set()

# Filter: Keep only files NOT in the training set
test_pool = [f for f in all_files if f not in training_set]
print("Testpool length:", len(test_pool))

if len(test_pool) < 5:
    print("Warning: Not enough test files found. Check your paths.")
    selected_files = all_files[:5]
else:
    # Pick 5 random UNSEEN files
    selected_files = np.random.choice(test_pool, size=10, replace=False)

print(f"Testing on {len(selected_files)} unseen files.")

# ==============================================================================
# 3. PREPARE MODEL & SCALERS
# ==============================================================================
# Load Model
model = PhysicsEmulator(input_dim=8).to(DEVICE)
model.load_state_dict(torch.load("Best_EOS_Model.pth", map_location=DEVICE))
model.eval()

# IMPORTANT: You must use the SAME mean/std from TRAINING.
# If you are in a new session, you should have saved these. 
# For now, assuming they exist in memory or you manually input them.
# Example: 
# X_eos_mean = torch.load("Datasets/X_eos_mean.pt")
# X_eos_std = torch.load("Datasets/X_eos_std.pt")



# if 'X_eos_mean' not in locals():
#     print("ERROR: X_eos_mean and X_eos_std are missing!")
#     print("You must save these during training or run this cell in the same notebook.")
#     # Loading the SAME mean/std from TRAINING.
#     X_eos_mean = torch.load("X_eos_mean.pt", map_location=DEVICE)
#     X_eos_std = torch.load("X_eos_std.pt", map_location=DEVICE)
# else:
#     X_eos_mean = X_eos_mean.to(DEVICE)
#     X_eos_std = X_eos_std.to(DEVICE)



X_eos_mean = torch.load("X_eos_mean.pt", map_location=DEVICE)
X_eos_std = torch.load("X_eos_std.pt", map_location=DEVICE)


# ==============================================================================
# 4. EVALUATION LOOP
# ==============================================================================
plt.figure(figsize=(10, 7))
colors = plt.cm.jet(np.linspace(0, 1, len(selected_files)))

for color, file_path in zip(colors, selected_files):
    filename = os.path.basename(file_path)
    eos_params = extract_eos_params(filename)
    
    if eos_params is None: continue

    # Load Ground Truth Data
    try:
        data = np.loadtxt(file_path)
    except: continue
        
    # Standard cleanup (Positive Mass/Radius only)
    mass_vals = data[:, 1]
    radius_vals = data[:, 2]
    
    # Filter stable branch (Up to Max Mass) same as training
    max_m_idx = np.argmax(mass_vals)
    # Optional: Apply the same safety margin cut as training
    cut_idx = max(1, max_m_idx - 2) 
    
    mass_vals = mass_vals[:cut_idx]
    radius_vals = radius_vals[:cut_idx]
    
    # ---------------------------------------------------------
    # PREPARE INPUT TENSORS (The Hybrid Normalization)
    # ---------------------------------------------------------
    
    # 1. Tile the EOS params for every mass point
    num_points = len(mass_vals)
    eos_repeated = np.tile(eos_params, (num_points, 1)) # Shape (N, 7)
    
    # 2. Convert to Tensor
    X_eos_input = torch.tensor(eos_repeated, dtype=torch.float32).to(DEVICE)
    X_mass_input = torch.tensor(mass_vals.reshape(-1, 1), dtype=torch.float32).to(DEVICE)
    
    # 3. NORMALIZE (Use training stats!)
    # EOS: Z-score
    X_eos_norm = (X_eos_input - X_eos_mean) / X_eos_std
    # Mass: Constant Scaling
    X_mass_norm = X_mass_input / MASS_SCALE
    
    # 4. Concatenate
    model_input = torch.cat((X_eos_norm, X_mass_norm), dim=1)
    
    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------
    with torch.no_grad():
        # Predict normalized Radius (0.0 to ~1.0)
        R_pred_norm = model(model_input)
        
        # De-normalize: R_phys = R_norm * 25.0
        R_pred_km = R_pred_norm * RADIUS_SCALE
        
        # Move to CPU for plotting
        R_pred_km = R_pred_km.cpu().numpy().flatten()

    # ---------------------------------------------------------
    # PLOT [m, L, J, n_v, d, B, n] # Label with Physics Params
    # ---------------------------------------------------------
    label_txt = f"""m={eos_params[0]:.2f}, L={eos_params[1]:.0f}, J={eos_params[2]:.0f}, n_v={eos_params[3]:.2f}, d={eos_params[4]:.2f}, B={eos_params[5]:.2f}, n={eos_params[6]:.3f}""" 
    
    # Plot Ground Truth (Solid Line)
    plt.plot(radius_vals, mass_vals, "-", color=color, alpha=0.5, linewidth=2)
    
    # Plot Prediction (Dashed Line) 
    plt.plot(R_pred_km, mass_vals, "--", color=color, linewidth=2, label=label_txt)

plt.title("Neural Network vs. TOV Solver (Unseen Files)")
plt.xlabel("Radius (km)")
plt.ylabel(r"Mass ($M_{\odot}$)")
plt.legend(title="Predictions", fontsize='small')
plt.grid(True, alpha=0.3)
plt.xlim(9, 20) # Focus on typical NS radius range
plt.ylim(0, 3.5)
plt.savefig("Testing plot 2.svg")

print("Testing complete. Plot saved to 'Testing plot.svg'.")