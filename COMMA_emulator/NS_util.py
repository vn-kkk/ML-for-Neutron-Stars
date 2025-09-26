import numpy as np
import eos
import tov
import tov_tide

dkm = 1.3234e-6 #conversion from Mev/fm^3 to km^-2
conv=197.33**3             #in MeV/fm3

# This function returns Energy and pressure array given given eos parameters
#and if eos is not montonic returns nan
def EoS(theta):
    L0,Ksym,rho1,rho2,rho3,Gamma1,Gamma2,Gamma3 = theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7]
    index, nb, pressure, energy, cs, icc = eos.eos(L0,Ksym,rho1,rho2,rho3,Gamma1,Gamma2,Gamma3)
    nb = nb[0:index-1]
    pressure = pressure[0:index-1]
    energy = energy[0:index-1]
    cs = cs[0:index-1]
    pres=pressure*dkm
    ener=energy*dkm
    return ener, pres, nb, cs, icc


def get_mass(theta,pcs):
    ms=[]
    for i in pcs:
        L0,Ksym,rho1,rho2,rho3,Gamma1,Gamma2,Gamma3 = theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7]
        index, nb, pressure, energy, cs, icc = eos.eos(L0,Ksym,rho1,rho2,rho3,Gamma1,Gamma2,Gamma3)
        pressure = pressure[0:index-1]
        energy = energy[0:index-1]
        pres=pressure*dkm
        ener=energy*dkm
        m,r = tov.tov(ener,pres,[i])
        ms.append(m)
    return ms

# Find the first maxima
def find_first_maxima(arr):
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            return i, arr[i]  # Return the index and the value of the first maxima
    max_index = np.argmax(arr)
    return max_index, arr[max_index]

def filter_mass_radius_td(eos_params, pcmin):
    """
    Filters the mass, radius, and tidal deformability arrays based on given EOS parameters
    and a minimum pressure constraint.
    
    Parameters:
        eos_params (list): Parameters for the equation of state (EOS).
        pcmin (float): Fiducial minimum pressure.

    Returns:
        m_array (np.ndarray): Filtered mass array.
        r_array (np.ndarray): Filtered radius array.
        td_array (np.ndarray): Filtered tidal deformability array.
        mmax (float): Maximum mass found.
        pcmax (float): Maximum pressure corresponding to the maximum mass.
    """
    # Generate EOS data
    ee, pp, nb, cs, icc = EoS(eos_params)
    # Check conditions
    if icc == 0 and max(pp) > pcmin:
        # Create pressure grid
        pc = np.logspace(np.log10(pcmin), np.log10(max(pp)), 50)
        pc[-1] = max(pp)
        
        # Get mass, radius, and tidal deformability
        m, r, td = get_MRL_pc(eos_params, pc)

        # Convert to NumPy arrays for element-wise operations
        m = np.array(m)
        r = np.array(r)
        td = np.array(td)

        # Find first maxima in mass array
        j, mmax = find_first_maxima(m)

        # Create mask to filter arrays where 0.9 < m and indices are up to j
        mask = (m > 0.9) & (np.arange(len(m)) <= j)

        # Apply the mask to the mass, radius, and tidal deformability arrays
        m_array = m[mask]
        r_array = r[mask]
        td_array = td[mask]

        # Calculate pcmax (maximum pressure corresponding to the maximum mass)
        pcmax = pc[j]  # You can modify this if you have a specific way to calculate pcmax
        if pcmax== max(pp):
            print("getting maximum mass at last point; reject this")
            return np.array([]), np.array([]), np.array([]), None, None
        else:
            return m_array, r_array, td_array, mmax, pcmax

    else:
        print("EOS check failed. Returning empty arrays.")
        return np.array([]), np.array([]), np.array([]), None, None
        

def get_MR_pc(theta,pc):
    M=[]
    R=[]
    L=[]
    for i in pc:
        L0,Ksym,rho1,rho2,rho3,Gamma1,Gamma2,Gamma3 = theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7]
        index, nb, pressure, energy, cs, icc = eos.eos(L0,Ksym,rho1,rho2,rho3,Gamma1,Gamma2,Gamma3)
        pressure = pressure[0:index-1]
        energy = energy[0:index-1]
        pres=pressure*dkm
        ener=energy*dkm
        m,r = tov.tov(ener,pres,[i])
        M.append(m)
        R.append(r)
    return M,R

def get_MRL_pc(theta,pc):
    M=[]
    R=[]
    L=[]
    for i in pc:
        L0,Ksym,rho1,rho2,rho3,Gamma1,Gamma2,Gamma3 = theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7]
        index, nb, pressure, energy, cs, icc = eos.eos(L0,Ksym,rho1,rho2,rho3,Gamma1,Gamma2,Gamma3)
        pressure = pressure[0:index-1]
        energy = energy[0:index-1]
        pres=pressure*dkm
        ener=energy*dkm
        m,r,td = tov_tide.tov_tide(ener,pres,[i])
        M.append(m)
        R.append(r)
        L.append(td)
    return M,R,L
