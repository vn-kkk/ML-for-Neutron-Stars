
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


################################################################################
# IMPORT ALL REQUIRED MODULES
################################################################################
import os
import re
import glob
import sys

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
 
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from numba import jit

import tov_tide


################################################################################
# GLOBAL CONSTANTS AND UNIT CONVERSION FACTORS
################################################################################
msun=147660                 # Solar mass in cm given by the formula G*M_sun/c^2

c=2.9979e10                 # speed of light in cm/s (CGS)
G=6.67408e-8                # gravitational constant in cm^3/gm/s^2 (CGS)

dkm = 1.3234e-06            # conversion of MeV/fm^3 to km^-2
dcgs = 1.78e12              # factor to convert from MeV/fm^3 to gm/cm^3
conv = 8.2601e-40           # dyn/cm^2 to km^-2
cgs1=1.7827e+12             # MeV/fm3 to gms/cm3
cgs2=1.6022e+33             # MeV/fm3 to dyne/cm2


################################################################################
# PIECEWISE POLYTROPIC LOW-DENSITY (CRUST) PARAMETERS 
################################################################################
# Polytropic exponents
GammaL_1 = 1.35692
GammaL_2 = 0.62223
GammaL_3 = 1.28733
GammaL_4 = 1.58425

# Polytropic constants
KL_1 = 3.99874e-8 # * pow(Msun/Length**3, GammaL_1-1)
KL_2 = 5.32697e+1 # * pow(Msun/Length**3, GammaL_2-1)
KL_3 = 1.06186e-6 # * pow(Msun/Length**3, GammaL_3-1)
KL_4 = 6.80110e-9 # * pow(Msun/Length**3, GammaL_4-1)
# notice a missing c^2 in Ki values in Table II of Read et al. 2009

# Densities at the boundaries of the piecewise polytropes
rhoL_1 = 2.62789e12
rhoL_2 = 3.78358e11
rhoL_3 = 2.44034e7
rhoL_4 = 0.0

# Pressures at the boundaries of the piecewise polytropes
pL_1 = KL_1*rhoL_1**GammaL_1
pL_2 = KL_2*rhoL_2**GammaL_2
pL_3 = KL_3*rhoL_3**GammaL_3
pL_4 = 0.0

# The exact numbers are taken from a particular crust model/table.


################################################################################
# FEW MORE CALCULATIONS
################################################################################
# Compute the offsets specific internal energy (epsL_i) and alphaL_i at the boundaries
# The general form used: ε(ρ)=(1+α)ρ+K/(Γ−1)ρ^Γ. Solving for alpha ensures matching across boundaries.
# Energy density needs an additive offset to enforce continuity.

epsL_4 = 0.0
alphaL_4 = 0.0
epsL_3 = (1+alphaL_4)*rhoL_3 + KL_4/(GammaL_4 - 1.)*pow(rhoL_3, GammaL_4)
alphaL_3 = epsL_3/rhoL_3 - 1.0 - KL_3/(GammaL_3 - 1.)*pow(rhoL_3, GammaL_3 -1.0)
epsL_2 = (1+alphaL_3)*rhoL_2 + KL_3/(GammaL_3 - 1.)*pow(rhoL_2, GammaL_3)
alphaL_2 = epsL_2/rhoL_2 - 1.0 - KL_2/(GammaL_2 - 1.)*pow(rhoL_2, GammaL_2 -1.0)
epsL_1 = (1+alphaL_2)*rhoL_1 + KL_2/(GammaL_2 - 1.)*pow(rhoL_1, GammaL_2)
alphaL_1 = epsL_1/rhoL_1 - 1.0 - KL_1/(GammaL_1 - 1.)*pow(rhoL_1, GammaL_1 -1.0)

# Density thresholds for high-density polytropes
rho1 = pow(10,14.7) # Break Density
rho2 = pow(10,15.0) # Break Density

# GR conversion prefactors to go from cgs pressure/energy-density units into geometric units (where G=c=1)
t_p=G/c**4
t_rho=G/c**2


################################################################################
# FORWARD EOS: FROM (logp, Gamma1, Gamma2, Gamma3) TO p(ρ) AND ε(ρ)
################################################################################
def p_eps_of_rho(rho,logp,Gamma1,Gamma2,Gamma3):
    p1 = pow(10.0,logp)/c**2
    K1 = p1/pow(rho1,Gamma1)
    K2 = K1 * pow( rho1, Gamma1-Gamma2)
    K3 = K2 * pow( rho2, Gamma2-Gamma3)
    rho0 = pow(KL_1/K1,1.0/(Gamma1-GammaL_1)) 
    eps0 = (1.0+alphaL_1)*rho0 + KL_1/(GammaL_1-1.0)*pow(rho0,GammaL_1)
    alpha1 = eps0/rho0 - 1.0 - K1/(Gamma1 - 1.0)*pow(rho0, Gamma1 -1.0)
    eps1 = (1.0+alpha1)*rho1 + K1/(Gamma1 - 1.0)*pow(rho1, Gamma1)
    alpha2 = eps1/rho1 - 1.0 - K2/(Gamma2 - 1.0)*pow(rho1, Gamma2 -1.0)
    eps2 = (1.0+alpha2)*rho2 + K2/(Gamma2 - 1.0)*pow(rho2, Gamma2)
    alpha3 = eps2/rho2 - 1.0 - K3/(Gamma3 - 1.0)*pow(rho2, Gamma3 -1.0)
    if rho<rhoL_3:
        p = KL_4*pow(rho,GammaL_4)
        eps = (1.0+alphaL_4)*rho + KL_4/(GammaL_4-1.0)*pow(rho,GammaL_4)
    elif rhoL_3<= rho <rhoL_2:
        p = KL_3*pow(rho,GammaL_3)
        eps = (1.0+alphaL_3)*rho + KL_3/(GammaL_3-1.0)*pow(rho,GammaL_3)
    elif rhoL_2<= rho <rhoL_1:
        p = KL_2*pow(rho,GammaL_2)
        eps = (1.0+alphaL_2)*rho + KL_2/(GammaL_2-1.0)*pow(rho,GammaL_2)
    elif rhoL_1<= rho <rho0:
        p = KL_1*pow(rho,GammaL_1)
        eps = (1.0+alphaL_1)*rho + KL_1/(GammaL_1-1.0)*pow(rho,GammaL_1)
    elif rho0<= rho <rho1:
        p = K1*pow(rho,Gamma1)
        eps = (1.0+alpha1)*rho + K1/(Gamma1-1.0)*pow(rho,Gamma1)
    elif rho1<= rho <rho2:
        p = K2*pow(rho,Gamma2)
        eps = (1.0+alpha2)*rho + K2/(Gamma2-1.0)*pow(rho,Gamma2)
    else:
        p = K3*pow(rho,Gamma3)
        eps = (1.0+alpha3)*rho + K3/(Gamma3-1.0)*pow(rho,Gamma3)
    return p*c**2, eps*c**2


################################################################################
# INVERSE EOS: FROM (logp, Gamma1, Gamma2, Gamma3) TO ε(p)
################################################################################
@jit(nopython=True)
def eps(p,logp,Gamma1,Gamma2,Gamma3):
    p1 = pow(10.0,logp)/c**2
    p*=1/c**2
    K1 = p1/pow(rho1,Gamma1)
    K2 = K1 * pow( rho1, Gamma1-Gamma2)
    K3 = K2 * pow( rho2, Gamma2-Gamma3)
    rho0 = pow(KL_1/K1,1.0/(Gamma1-GammaL_1))
    eps0 = (1.0+alphaL_1)*rho0 + KL_1/(GammaL_1-1.0)*pow(rho0,GammaL_1)
    alpha1 = eps0/rho0 - 1.0 - K1/(Gamma1 - 1.0)*pow(rho0, Gamma1 -1.0)
    eps1 = (1.0+alpha1)*rho1 + K1/(Gamma1 - 1.0)*pow(rho1, Gamma1)
    alpha2 = eps1/rho1 - 1.0 - K2/(Gamma2 - 1.0)*pow(rho1, Gamma2 -1.0)
    eps2 = (1.0+alpha2)*rho2 + K2/(Gamma2 - 1.0)*pow(rho2, Gamma2)
    alpha3 = eps2/rho2 - 1.0 - K3/(Gamma3 - 1.0)*pow(rho2, Gamma3 -1.0)
    p0 = K1*pow(rho0,Gamma1)
    p2 = K3*pow(rho2,Gamma3)
    if  p<pL_3:
        rho = pow(p/KL_4,1/GammaL_4)
        eps = (1.0+alphaL_4)*rho + KL_4/(GammaL_4-1.0)*pow(rho,GammaL_4)
    elif pL_3<= p <pL_2:
        rho = pow(p/KL_3,1/GammaL_3)
        eps = (1.0+alphaL_3)*rho + KL_3/(GammaL_3-1.0)*pow(rho,GammaL_3)
    elif pL_2<= p <pL_1:
        rho = pow(p/KL_2,1/GammaL_2)
        eps = (1.0+alphaL_2)*rho + KL_2/(GammaL_2-1.0)*pow(rho,GammaL_2)
    elif  pL_1<p <p0:
        rho = pow(p/KL_1,1/GammaL_1)
        eps = (1.0+alphaL_1)*rho + KL_1/(GammaL_1-1.0)*pow(rho,GammaL_1)
    elif p0<= p <p1:
        rho = pow(p/K1,1/Gamma1)
        eps = (1.0+alpha1)*rho + K1/(Gamma1-1.0)*pow(rho,Gamma1)
    elif p1<= p <p2:
        rho = pow(p/K2,1/Gamma2)
        eps = (1.0+alpha2)*rho + K2/(Gamma2-1.0)*pow(rho,Gamma2)
    else:
        rho = pow(p/K3,1/Gamma3)
        eps = (1.0+alpha3)*rho + K3/(Gamma3-1.0)*pow(rho,Gamma3)
    return eps*c**2


################################################################################
# THE TOV INTEGRATOR
################################################################################
def TOV(logrho_c, theta, compute_tidal=True):
    logp, Gamma1, Gamma2, Gamma3 = theta
    dr = 100.0

    rho_c = 10**logrho_c
    r = 0.1
    m = 0.0

    p, e = p_eps_of_rho(rho_c, logp, Gamma1, Gamma2, Gamma3)
    p *= t_p
    e *= t_p

    # --- store profiles ---
    p_prof = []
    e_prof = []
    r_prof = []
    m_prof = []

    while p > 0:
        p_prof.append(p)
        e_prof.append(e)
        r_prof.append(r)
        m_prof.append(m)

        dp = -(e + p) * (m + 4*np.pi*r**3*p) / (r*(r - 2*m))
        p += dp * dr
        if p <= 0:
            break

        m += 4*np.pi*r**2 * e * dr
        r += dr
        e = eps(p/t_p, logp, Gamma1, Gamma2, Gamma3) * t_p

    # final mass and radius
    M = m / msun
    R = r / 1e5

    if not compute_tidal:
        return M, R

    # --- prepare inputs for Fortran ---
    p_prof = np.array(p_prof, dtype=np.float64)
    e_prof = np.array(e_prof, dtype=np.float64)

    # Fortran expects central pressure at index N
    p_prof = p_prof[::-1]
    e_prof = e_prof[::-1]

    pc = p_prof[-1]
    N = len(p_prof)

    # --- tidal deformability ---
    M_tide, R_tide, Lambda = tov_tide.tov_tide(
        e_prof,
        p_prof,
        pc
    )
    
    return M, R, Lambda # Reyrns true lambda (not log lambda)


################################################################################
# CREATE DATASET FOR TESTING THE MODEL
################################################################################
# Number of EOS samples
NUM_SAMPLES = 40000
# Directory to save/load dataset and models 
save_dir = f"{NUM_SAMPLES}files"


EOS_params = np.random.uniform( low=[1.4, 1.4, 1.4], 
                                high=[5., 5., 5.], 
                                size=(NUM_SAMPLES, 3)
                                )

logrho_c_samples = np.random.uniform(14.5, 15.4, size=(NUM_SAMPLES, 1))
logp_samples = np.random.uniform(33.5, 34.8, size=(NUM_SAMPLES, 1))

MRL_data = []

for i in trange(NUM_SAMPLES, desc="Solving TOV"):
    logrho_c = logrho_c_samples[i, 0]
    logp = logp_samples[i, 0]
    params = EOS_params[i]

    M, R, Lambda = TOV(
        logrho_c,
        [logp, params[0], params[1], params[2]],
        compute_tidal = True
        )
    MRL_data.append([M, R, Lambda])

MRL_data = np.array(MRL_data)

# ==========================================================
# FILTER UNPHYSICAL OUTPUTS PRODUCED BY EXTREMELY STIFF EOSs
# ==========================================================

M = MRL_data[:, 0]
R = MRL_data[:, 1]
Lambda = MRL_data[:, 2]

mask = (
    np.isfinite(M) &
    np.isfinite(R) &
    (M > 0.15) & (M < 3.5) &
    (R > 6.0) & (R < 25.0) &
    (Lambda > 0) & (Lambda < 1e6)
)

EOS_data = np.hstack([
    logrho_c_samples[mask],
    logp_samples[mask],
    EOS_params[mask],
    MRL_data[mask]
])

print(f"Kept {EOS_data.shape[0]} / {NUM_SAMPLES} samples")

# Save cleaned dataset
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, f"EOS_dataset_{NUM_SAMPLES}files.npy"), EOS_data)
print("Datasets created and saved!")


################################################################################
# LOAD AND PREP THE DATA
################################################################################
# Load dataset
data = np.load(os.path.join(save_dir, f"EOS_dataset_{NUM_SAMPLES}files.npy"))

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
BATCH_SIZE = 256

# Split into training and validation sets
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

# Convert to Tensor
# Inputs: Cols 0 to 4 (5 features: log_rho_c, log_p, Gamma1, Gamma2, Gamma3)
# Output: Col 5 - 7 (Mass, Radius, Tidal Deformability)
X_eos_train = torch.tensor(train_data[:, :5], dtype=torch.float32)
y_eos_train = torch.tensor(train_data[:, 5:8], dtype=torch.float32)

X_eos_val = torch.tensor(val_data[:, :5], dtype=torch.float32)
y_eos_val = torch.tensor(val_data[:, 5:8], dtype=torch.float32)

# Normalize using Z-Score on all the inputs
X_eos_mean = X_eos_train.mean(dim=0, keepdim=True)
X_eos_std = X_eos_train.std(dim=0, keepdim=True)

# Save them:
torch.save(X_eos_mean, os.path.join(save_dir, "X_eos_mean.pt"))
torch.save(X_eos_std, os.path.join(save_dir, "X_eos_std.pt"))
print("Normalization statistics saved.")

X_train_norm = (X_eos_train - X_eos_mean) / X_eos_std
X_val_norm = (X_eos_val - X_eos_mean) / X_eos_std

# 4. Separate Mass, Radius and TD
y_mass_train, y_radius_train, y_td_train = y_eos_train[:, 0:1], y_eos_train[:, 1:2], y_eos_train[:, 2:3]
y_mass_val, y_radius_val, y_td_val = y_eos_val[:, 0:1], y_eos_val[:, 1:2], y_eos_val[:, 2:3]

# 5.1  Constant Scaling on Mass (M)
MASS_SCALE = 3.5
y_mass_train_norm = y_mass_train / MASS_SCALE
y_mass_val_norm = y_mass_val / MASS_SCALE

# 5.2. Normalize using Constant Scaling on Radius (R)
RADIUS_SCALE = 25.0
y_radius_train_norm = y_radius_train / RADIUS_SCALE
y_radius_val_norm = y_radius_val / RADIUS_SCALE

# 5.3 Log Scale TD
y_td_train_norm = torch.log10(y_td_train)
y_td_val_norm = torch.log10(y_td_val)

# 6. --- Recombine Outputs ---
y_train_norm = torch.cat((y_mass_train_norm, y_radius_train_norm, y_td_train_norm), dim=1)
y_val_norm = torch.cat((y_mass_val_norm, y_radius_val_norm, y_td_val_norm), dim=1)



################################################################################
# DEFINE THE MODEL
################################################################################
class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim, auxiliary_dim=1):
        super().__init__()
        # We accept the hidden state + the auxiliary Central Pressure injection
        self.fc = nn.Linear(hidden_dim + auxiliary_dim, hidden_dim)
        self.act = nn.GELU() # Gaussian Error Linear Unit
    
    def forward(self, x, cp):
        # Concatenate Central Pressure to the input of the layer
        combined = torch.cat([x, cp], dim=1)
        out = self.act(self.fc(combined))
        return x + out # Residual connection

class PhysicsEmulator(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=512): 
        super().__init__()
        # Separate EOS inputs from Central Pressure
        # input_dim is 5: (4 EOS params + 1 Central Pressure)
        self.eos_dim = input_dim - 1 
        
        # 1. Initial encoding of EOS parameters only: 
        self.input_layer = nn.Linear(self.eos_dim, hidden_dim)
        
        # 2. Deep Residual Layers with Central Pressure Injection
        self.block1 = ResNetBlock(hidden_dim, auxiliary_dim=1)
        self.block2 = ResNetBlock(hidden_dim, auxiliary_dim=1)
        self.block3 = ResNetBlock(hidden_dim, auxiliary_dim=1)
        self.block4 = ResNetBlock(hidden_dim, auxiliary_dim=1)

        # 3. Output layers
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2), # Inject Central Pressure one last time
            nn.GELU(),                                      # 513 --> 256
            nn.Linear(hidden_dim // 2, 3)                   # Output: 256 --> 3
        )
        
    def forward(self, x):
        # Split input into EOS params and Central Pressure
        # Central Pressure is the second column
        cp = x[:, 1:2]
        # EOS parameters are the remaining columns
        eos_params = torch.cat([x[:, 0:1], x[:, 2:]], dim=1)

        # 1. Encode EOS
        x_hidden = self.input_layer(eos_params)             # Input: 5 --> 512

        # 2. Central Pressure-Injected Residual Flow: Pass through blocks, injecting Central Pressure at each step
        x_hidden = self.block1(x_hidden, cp)              # 512 + 1 --> 512
        x_hidden = self.block2(x_hidden, cp)              # 512 + 1 --> 512
        x_hidden = self.block3(x_hidden, cp)              # 512 + 1 --> 512
        x_hidden = self.block4(x_hidden, cp)              # 512 + 1 --> 512
        
        # 3. Final Prediction
        # Concatenate Central Pressure one last time for the read-out
        combined_final = torch.cat([x_hidden, cp], dim=1) # 512 + 1 = 513
        return self.final_layer(combined_final)             
    

################################################################################
# TRAIN THE MODEL
################################################################################
# --- Helper function for plotting ---
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
        plt.savefig(os.path.join(save_dir, filename))
        print(f"Loss plot saved to {filename}", flush=True)
    except Exception as e:
        print(f"ERROR saving plot: {e}", flush=True)
    plt.close()
# --------------------------------------------------------------------------

model = PhysicsEmulator().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-7)
criterion = nn.HuberLoss()

# ==============================================================================
# 5. TRAINING
# ==============================================================================
# Ensure Mass and Radius are Torch Tensors (it might be a numpy array currently)
if isinstance(y_train_norm, np.ndarray):
    y_train_norm = torch.from_numpy(y_train_norm).to(torch.float32)
if isinstance(y_val_norm, np.ndarray):
    y_val_norm = torch.from_numpy(y_val_norm).to(torch.float32)

# Update your DataLoaders
train_loader = DataLoader(TensorDataset(X_train_norm, y_train_norm), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_norm, y_val_norm), batch_size=BATCH_SIZE, shuffle=False)
epochs = 500

best_val_loss = float('inf')
patience = 50  # Number of epochs to wait for improvement before stopping
patience_counter = 0

# 1. ADD STORAGE LISTS
train_losses = []
val_losses = []
# Ensure MASS_SCALE and RADIUS_SCALE are defined globally or passed in if running as a function
# If you didn't define save_dir, model will save in current directory.

for epoch in range(epochs):
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

    # Early Stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model weights
        best_model_state = model.state_dict()

    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if epoch % 100 == 0:
        # Calculate the Approximate Physical Error in km
        radius_error_km = np.sqrt(2 * val_loss) * RADIUS_SCALE 
        mass_error = np.sqrt(2 * val_loss) * MASS_SCALE
        td_error_log = np.sqrt(2 * val_loss)  # Since td is log-scaled, this is in log units


        print(f"Epoch {epoch} | Train Loss: {train_loss:.6e} | Val Loss: {val_loss:.6e} | Approx Radius Error: {radius_error_km:.4f} km | Approx Mass Error: {mass_error:.4f} | Approx TD error: {td_error_log:.4f}", flush=True)

        # 3. PLOT AND SAVE PERIODICALLY
        # Plot every 250 epochs (or choose a different interval)
        if epoch % 250 == 0 and epoch > 0:
            plot_and_save_losses(train_losses, val_losses, filename=f"loss_curve_epoch{epoch}.png")

# Restore best model
model.load_state_dict(best_model_state)
print(f"Training finished. Best validation loss: {best_val_loss:.10f}")

# 4. FINAL PLOT after training finishes
plot_and_save_losses(train_losses, val_losses, filename="loss_curve_final.png")
torch.save(model.state_dict(), os.path.join(save_dir, "Best_EOS_Model.pth"))