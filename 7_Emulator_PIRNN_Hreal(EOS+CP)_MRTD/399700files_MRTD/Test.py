import os
import re
import torch

import numpy as np

import torch.nn as nn
import glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================================================================
# 1. DEFINE THE MODEL
# ==============================================================================
class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim, auxiliary_dim=1):
        super().__init__()
        # We accept the hidden state + the auxiliary cp injection
        self.fc = nn.Linear(hidden_dim + auxiliary_dim, hidden_dim)
        self.act = nn.GELU() 
    
    def forward(self, x, cp):
        # Concatenate cp to the input of the layer
        combined = torch.cat([x, cp], dim=1)
        out = self.act(self.fc(combined))
        return x + out # Residual connection

class PhysicsEmulator(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=512): # Increased width
        super().__init__()
        # Separate EOS inputs from cp
        # input_dim is 8: (7 EOS params + 1 cp)
        self.eos_dim = input_dim - 1 
        
        # Initial encoding of EOS parameters only
        self.input_layer = nn.Linear(self.eos_dim, hidden_dim)
        
        # Deep Residual Layers with cp Injection
        self.block1 = ResNetBlock(hidden_dim, auxiliary_dim=1)
        self.block2 = ResNetBlock(hidden_dim, auxiliary_dim=1)
        self.block3 = ResNetBlock(hidden_dim, auxiliary_dim=1)
        self.block4 = ResNetBlock(hidden_dim, auxiliary_dim=1)
        
        # Output layers
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2), # Inject cp one last time
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)               # Output: 256 --> 3
        )
        
    def forward(self, x):
        # Split input into EOS params and Central Pressure
        # CP is the LAST column (index -1)
        eos_params = x[:, :-1]
        cp = x[:, -1:]
        
        # 1. Encode EOS
        x_hidden = self.input_layer(eos_params)
        
        # 2. Pass through blocks, injecting cp at each step
        x_hidden = self.block1(x_hidden, cp)
        x_hidden = self.block2(x_hidden, cp)
        x_hidden = self.block3(x_hidden, cp)
        x_hidden = self.block4(x_hidden, cp)
        
        # 3. Final Prediction
        # Concatenate cp one last time for the read-out
        combined_final = torch.cat([x_hidden, cp], dim=1)
        return self.final_layer(combined_final)

# ==============================================================================
# 2. CONFIGURATION & HELPERS
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RADIUS_SCALE = 25.0
MASS_SCALE = 3.5
NUM_FILES_TO_USE = 399700

save_dir = f"{NUM_FILES_TO_USE}files_MRTD"

N_MR_FILES = 10          # for Mass–Radius plot
N_TD_FILES = 1           # for Compactness–Tidal plot

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
# 3. SELECT "UNSEEN" FILES
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
    training_set = {
    os.path.basename(f) for f in training_files_used        
    }
except FileNotFoundError:
    print("Warning: List of training files not found. Using random files (risk of data leakage).")
    training_set = {
    os.path.basename(f) for f in training_files_used        
    }

# Filter: Keep only files NOT in the training set
test_pool = [
    f for f in all_files
    if os.path.basename(f) not in training_set
]
print("Total unseen files available for testing:", len(test_pool))

# Ensure we have enough unseen files
if len(test_pool) < max(N_MR_FILES, N_TD_FILES):
    raise RuntimeError("Not enough unseen files available.")

selected_files_MR = np.random.choice(
    test_pool, size=N_MR_FILES, replace=False
)
selected_files_TD = np.random.choice(
    test_pool, size=N_TD_FILES, replace=False
)
print(f"MR plot   : {len(selected_files_MR)} unseen files")
print(f"TD plot   : {len(selected_files_TD)} unseen files")

# ==============================================================================
# 4. PREPARE MODEL & SCALERS
# ==============================================================================
# Load Model
model = PhysicsEmulator(input_dim=8).to(DEVICE)
model.load_state_dict(torch.load("Best_EOS_Model.pth", map_location=DEVICE))
model.eval()

# IMPORTANT: You must use the SAME mean/std from TRAINING.
# You must save these during training or run this cell in the same notebook.
# Loading the SAME mean/std from TRAINING.
X_eos_mean = torch.load("X_eos_mean.pt", map_location=DEVICE)
X_eos_std = torch.load("X_eos_std.pt", map_location=DEVICE)


# ==============================================================================
# 5. EVALUATION LOOP
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(8, 6))
fig2, ax2 = plt.subplots(figsize=(8, 6))

# ------------------------------------------------------------------------------
# M-R plot loop
# ------------------------------------------------------------------------------

olors_MR = plt.cm.jet(np.linspace(0, 1, len(selected_files_MR)))

for color, file_path in zip(colors_MR, selected_files_MR):

    filename = os.path.basename(file_path)
    eos_params = extract_eos_params(filename)
    if eos_params is None:
        continue

    # load data
    try:
        data = np.loadtxt(file_path)
    except Exception:
        continue

    # get cp, mass and radius values
    cp_vals     = data[:, 0]    
    mass_vals   = data[:, 1]
    radius_vals = data[:, 2]

    # Fix to only use stable branch up to M_max - 2 points
    max_m_idx = np.argmax(mass_vals)
    cut_idx = max(1, max_m_idx - 2)

    cp_vals     = cp_vals[:cut_idx]
    mass_vals   = mass_vals[:cut_idx]
    radius_vals = radius_vals[:cut_idx]

    # Basic filtering to ensure positive values
    valid_mask = (radius_vals > 0) & (mass_vals > 0)

    cp_vals     = cp_vals[valid_mask]
    mass_vals   = mass_vals[valid_mask]
    radius_vals = radius_vals[valid_mask]


    # ---------------------------------------------------------
    # PREPARE INPUT TENSORS (The Hybrid Normalization)
    # ---------------------------------------------------------
    # 1. Tile the EOS params for every mass point
    num_points = len(mass_vals)
    eos_repeated = np.tile(eos_params, (num_points, 1))

    # 2. Convert to Tensor
    X_eos_input  = torch.tensor(eos_repeated, dtype=torch.float32, device=DEVICE)
    X_cp_input = torch.tensor(cp_vals[:, None], dtype=torch.float32, device=DEVICE)

    # 3. NORMALIZE (Use training stats!)
    # EOS: Z-scor
    X_eos_norm  = (X_eos_input - X_eos_mean) / X_eos_std
    # CP: Log Scaling
    X_cp_norm = torch.log10(X_cp_input)

    # 4. Concatenate inputs
    model_input = torch.cat((X_eos_norm, X_cp_norm), dim=1)

    # ---------------------------------------------------------
    # PREDICT AND DE-NORMALIZE
    # ---------------------------------------------------------
    with torch.no_grad():
        pred = model(model_input)
        M_pred = (pred[:, 0] * MASS_SCALE).cpu().numpy()
        R_pred_km = (pred[:, 1] * RADIUS_SCALE).cpu().numpy()

    label = (
        f"m={eos_params[0]:.2f}, L={eos_params[1]:.0f}, "
        f"J={eos_params[2]:.0f}, n_v={eos_params[3]:.2f}"
    )

    # ---------------------------------------------------------
    # PLOT [m, L, J, n_v, d, B, n] # Label with Physics Params
    # ---------------------------------------------------------
    ax1.plot(radius_vals, mass_vals, "-", color=color, alpha=0.4)
    ax1.plot(R_pred_km, M_pred, "--", color=color, label=label)

    colors_TD = plt.cm.viridis(np.linspace(0, 1, len(selected_files_TD)))


# ------------------------------------------------------------------------------
# Tidal Deformability plot loop
# ------------------------------------------------------------------------------

for color, file_path in zip(colors_TD, selected_files_TD):

    filename = os.path.basename(file_path)
    eos_params = extract_eos_params(filename)
    if eos_params is None:
        continue

    # load data
    try:
        data = np.loadtxt(file_path)
    except Exception:
        continue

    # get cp, mass, radius and tidal deformability values
    cp_vals     = data[:, 0]
    mass_vals   = data[:, 1]
    radius_vals = data[:, 2]
    td_vals     = data[:, 3]

    # Filter stable branch (Up to Max Mass) same as training
    max_m_idx = np.argmax(mass_vals)
    cut_idx = max(1, max_m_idx - 2)

    cp_vals     = cp_vals[:cut_idx]
    mass_vals   = mass_vals[:cut_idx]
    radius_vals = radius_vals[:cut_idx]
    td_vals     = td_vals[:cut_idx]

    # Basic filtering to ensure positive values
    valid_mask = (radius_vals > 0) & (mass_vals > 0)

    cp_vals     = cp_vals[valid_mask]
    mass_vals   = mass_vals[valid_mask]
    radius_vals = radius_vals[valid_mask]
    td_vals     = td_vals[valid_mask]

    # ---------------------------------------------------------
    # PREPARE INPUT TENSORS (The Hybrid Normalization)
    # ---------------------------------------------------------
    # 1. Tile the EOS params for every mass point
    num_points = len(mass_vals)

    eos_repeated = np.tile(eos_params, (num_points, 1))
    X_eos_input  = torch.tensor(eos_repeated, dtype=torch.float32, device=DEVICE)
    X_cp_input   = torch.tensor(cp_vals[:, None], dtype=torch.float32, device=DEVICE)

    X_eos_norm  = (X_eos_input - X_eos_mean) / X_eos_std
    X_cp_norm   = torch.log10(X_cp_input)

    model_input = torch.cat((X_eos_norm, X_cp_norm), dim=1)

    # ---------------------------------------------------------
    # PREDICT AND DE-NORMALIZE
    # ---------------------------------------------------------
    with torch.no_grad():
        pred = model(model_input)
        M_pred = (pred[:, 0] * MASS_SCALE).cpu().numpy()
        R_pred_km = (pred[:, 1] * RADIUS_SCALE).cpu().numpy()
        td_pred   = (10 ** pred[:, 2]).cpu().numpy()

    compact_truth = mass_vals / radius_vals
    compact_pred  = M_pred / R_pred_km

    # ---------------------------------------------------------
    # PLOT [m, L, J, n_v, d, B, n] # Label with Physics Params
    # ---------------------------------------------------------
    ax2.plot(compact_truth, np.log10(td_vals), "-", color=color, alpha=0.4)
    ax2.plot(compact_pred,  np.log10(td_pred), "--", color=color)

    # Formatting
    ax1.set_title("Mass–Radius Relations (Unseen EOS)")
    ax1.set_xlabel("Radius (km)")
    ax1.set_ylabel(r"Mass ($M_\odot$)")
    ax1.set_xlim(9, 20)
    ax1.set_ylim(0, 3.5)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize="small")

    ax2.set_title("Tidal Deformability vs Compactness")
    ax2.set_xlabel(r"Compactness ($M/R$)")
    ax2.set_ylabel("Tidal Deformability (log$_{10}$)")
    ax2.grid(alpha=0.3)

fig1.tight_layout()
fig2.tight_layout()

fig1.savefig(os.path.join(save_dir, "MR_unseen.png"))
fig2.savefig(os.path.join(save_dir, "TD_compactness_unseen.png"))
plt.show()

# ==============================================================================
# 5. COMPUTE METRICS OVER ALL POINTS AND SAVE TO FILE
# ==============================================================================
orig_radius_all = radius_vals
pred_radius_all = R_pred_km

orig_mass_all = mass_vals
pred_mass_all = M_pred

orig_td_all = td_vals
pred_td_all = td_pred

mae_radius = mean_absolute_error(orig_radius_all, pred_radius_all)
rmse_radius = np.sqrt(mean_squared_error(orig_radius_all, pred_radius_all))
r2_radius = r2_score(orig_radius_all, pred_radius_all)

mae_mass = mean_absolute_error(orig_mass_all, pred_mass_all)
rmse_mass = np.sqrt(mean_squared_error(orig_mass_all, pred_mass_all))
r2_mass = r2_score(orig_mass_all, pred_mass_all)

mae_td = mean_absolute_error(orig_td_all, pred_td_all)
rmse_td = np.sqrt(mean_squared_error(orig_td_all, pred_td_all))
r2_td = r2_score(orig_td_all, pred_td_all)

norm_rmse = np.mean([
    rmse_radius / np.mean(orig_radius_all),
    rmse_mass / np.mean(orig_mass_all),
    rmse_td / np.mean(orig_td_all)
])

print(f"Radius:  MAE={mae_radius:.4f}, RMSE={rmse_radius:.4f}, R²={r2_radius:.4f}")
print(f"Mass:  RMSE={rmse_mass:.4f}, R²={r2_mass:.4f}")
print(f"Tidal Deformability: RMSE={rmse_td:.4f}, R²={r2_td:.4f}")
print(f"Combined normalized RMSE = {norm_rmse:.4f}")

# -----------------------------
# Save metrics to a text file
# -----------------------------
output_file = os.path.join(save_dir, "Model_metrics.txt")

with open(output_file, "w") as f:
    f.write("Mass, Radius and TD prediction metrics\n")
    f.write("======================================\n")

    f.write("Mass Prediction Metrics:\n")
    f.write(f"MAE   : {mae_mass:.6f}\n")
    f.write(f"RMSE  : {rmse_mass:.6f}\n")
    f.write(f"R^2   : {r2_mass:.6f}\n")
    f.write("\n")

    f.write("Radius Prediction Metrics:\n")
    f.write(f"MAE   : {mae_radius:.6f}\n")
    f.write(f"RMSE  : {rmse_radius:.6f}\n")
    f.write(f"R^2   : {r2_radius:.6f}\n")
    f.write("\n")

    f.write("Tidal Deformability Prediction Metrics:\n")
    f.write(f"MAE   : {mae_td:.6f}\n")
    f.write(f"RMSE  : {rmse_td:.6f}\n")
    f.write(f"R^2   : {r2_td:.6f}\n")
    f.write("\n")

    f.write(f"Norm RMSE : {norm_rmse:.6f}\n")

print(f"Metrics saved to {output_file}")