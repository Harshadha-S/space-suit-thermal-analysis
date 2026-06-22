import numpy as np
import matplotlib.pyplot as plt

print("\n" + "="*75)
print("SPACE SUIT THERMAL ANALYSIS - LEO CONDITIONS")
print("="*75 + "\n")

# -----------------------------
# EDGE CASE SWITCH
# -----------------------------
CASE = "normal"  # "normal", "extreme_hot", "extreme_cold"

# -----------------------------
# MATERIAL PROPERTIES
# -----------------------------
materials = {
    'Kevlar': {'k': 0.04, 'rho': 1440, 'Cp': 1000, 'thickness': 0.003},
    'Nomex': {'k': 0.05, 'rho': 1300, 'Cp': 1050, 'thickness': 0.002},
    'MLI':   {'k': 0.0003, 'rho': 40, 'Cp': 1500, 'thickness': 0.010}
}

layer_sequence = ['Kevlar', 'Nomex', 'MLI']
thicknesses = np.array([materials[m]['thickness'] for m in layer_sequence])
total_thickness = np.sum(thicknesses)
layer_positions = np.concatenate(([0], np.cumsum(thicknesses)))

# -----------------------------
# GRID
# -----------------------------
nx = 160
x_grid = np.linspace(0, total_thickness, nx)
dx = total_thickness / (nx - 1)

k_grid = np.zeros(nx)
rho_grid = np.zeros(nx)
cp_grid = np.zeros(nx)

for i, x in enumerate(x_grid):
    for j, mat in enumerate(layer_sequence):
        if x <= layer_positions[j + 1]:
            k_grid[i] = materials[mat]['k']
            rho_grid[i] = materials[mat]['rho']
            cp_grid[i] = materials[mat]['Cp']
            break

alpha_grid = k_grid / (rho_grid * cp_grid)

# -----------------------------
# BOUNDARY CONDITIONS
# -----------------------------
T_inner_bc = 310.15
T_initial = 275 + 273.15

orbit_period = 90 * 60
sunlight_duration = 45 * 60

solar_constant = 1367
earth_ir = 50

sigma = 5.67e-8
epsilon = 0.85
alpha_solar = 0.7

def solar_flux(t):
    if CASE == "extreme_hot": return 1500
    if CASE == "extreme_cold": return 0
    return solar_constant if (t % orbit_period) < sunlight_duration else earth_ir

def net_flux(T, t):
    return alpha_solar * solar_flux(t) - epsilon * sigma * T**4

# -----------------------------
# TIME SETUP
# -----------------------------
dt = 25
total_time = 2 * orbit_period
num_steps = int(total_time / dt) + 1
time_array = np.linspace(0, total_time, num_steps)

alpha_mean = np.mean(alpha_grid)
r = dt * alpha_mean / (2 * dx**2)

# MATRICES
A = np.diag(1 + 2*r + np.zeros(nx)) + np.diag(-r*np.ones(nx-1),1) + np.diag(-r*np.ones(nx-1),-1)
B = np.diag(1 - 2*r + np.zeros(nx)) + np.diag(r*np.ones(nx-1),1) + np.diag(r*np.ones(nx-1),-1)

T = np.full(nx, T_initial)

T_outer, T_inner = [], []
heat_flux = []

print("Running simulation...")

for step in range(num_steps - 1):
    t = time_array[step]

    Q = net_flux(T[0], t)
    rhs = B @ T
    rhs[0] += 2*r*Q*dx / k_grid[0]

    A_bc = A.copy()
    A_bc[-1,:] = 0
    A_bc[-1,-1] = 1
    rhs[-1] = T_inner_bc

    T = np.linalg.solve(A_bc, rhs)

    T_outer.append(T[0])
    T_inner.append(T[-1])
    heat_flux.append(-k_grid[0]*(T[1]-T[0])/dx)

T_outer_C = np.array(T_outer) - 273.15
T_inner_C = np.array(T_inner) - 273.15
time_min = time_array[:-1]/60

# ============================================================================
# RADIATION MODEL
# ============================================================================
print("\nRADIATION ANALYSIS\n")

# NOTE:
# Simplified photon-like attenuation model (not full GCR physics)

configs = {
    'Kevlar only': ['Kevlar'],
    'Nomex only': ['Nomex'],
    'MLI only': ['MLI'],
    'All layers': ['Kevlar','Nomex','MLI']
}

radiation_data = {}
for m in materials:
    thickness_cm = materials[m]['thickness']*100
    density = materials[m]['rho']/1000
    radiation_data[m] = {'areal': thickness_cm*density}

energy_cases = {
    "Low": 0.08,
    "Medium": 0.05,
    "High": 0.02
}

def attenuation(layers, mu):
    tau = sum(mu * radiation_data[l]['areal'] for l in layers)
    return (1 - np.exp(-tau))*100

results = {e:{} for e in energy_cases}

for e, mu in energy_cases.items():
    for c, layers in configs.items():
        results[e][c] = attenuation(layers, mu)

selected_energy = "Medium"

# ============================================================================
# PLOTTING
# ============================================================================
fig, axs = plt.subplots(2,2, figsize=(12,8))

# TEMP
axs[0,0].plot(time_min, T_outer_C, label="Outer")
axs[0,0].plot(time_min, T_inner_C, label="Inner")
axs[0,0].legend()
axs[0,0].set_title("Temperature")

# HEAT FLUX
axs[0,1].plot(time_min, heat_flux)
axs[0,1].set_title("Heat Flux")

# RADIATION
names = list(results[selected_energy].keys())
vals = list(results[selected_energy].values())
bars = axs[1,0].bar(names, vals)

for b,v in zip(bars, vals):
    axs[1,0].text(b.get_x()+b.get_width()/2, v, f"{v:.1f}%", ha='center')

axs[1,0].set_title("Radiation")

# EMPTY
axs[1,1].axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\nSUMMARY\n")

best = max(results[selected_energy], key=results[selected_energy].get)

print(f"Max Temp: {max(T_outer_C):.2f} °C")
print(f"Best Radiation Config: {best} ({results[selected_energy][best]:.2f}%)")

print("\nObservations:")
print("- MLI dominates thermal insulation")
print("- Layering improves radiation shielding")
print("- Shielding effectiveness decreases with energy")
