import numpy as np
import matplotlib.pyplot as plt

print("\n" + "="*75)
print("SPACE SUIT THERMAL ANALYSIS - LEO CONDITIONS")
print("="*75 + "\n")

CASE = "normal" 
materials = {
    'Kevlar': {
        'k': 0.04,
        'rho': 1440,
        'Cp': 1000,
        'thickness': 0.003,
        'description': 'Para-aramid fiber (outer protection layer)'
    },
    'Nomex': {
        'k': 0.05,
        'rho': 1300,
        'Cp': 1050,
        'thickness': 0.002,
        'description': 'Meta-aramid insulator (structural layer)'
    },
    'MLI': {
        'k': 0.0003,
        'rho': 40,
        'Cp': 1500,
        'thickness': 0.010,
        'description': 'Multi-layer insulation (effective thermal barrier)'
    }
}
layer_sequence = ['Kevlar', 'Nomex', 'MLI']
thicknesses = np.array([materials[mat]['thickness'] for mat in layer_sequence])
total_thickness = np.sum(thicknesses)
layer_positions = np.concatenate(([0], np.cumsum(thicknesses)))

print("MATERIAL STACK (outer to inner):")
for i, mat in enumerate(layer_sequence):
    info = materials[mat]
    print(f"  Layer {i+1}: {mat:10s} | {info['thickness']*1000:5.1f} mm | "
          f"k={info['k']:.5f} W/m·K | ρ={info['rho']} kg/m³ | Cp={info['Cp']} J/kg·K")
print(f"  Total thickness: {total_thickness*1000:.1f} mm\n")


nx = 160
x_grid = np.linspace(0, total_thickness, nx)
dx = total_thickness / (nx - 1)

k_grid = np.zeros(nx)
rho_grid = np.zeros(nx)
cp_grid = np.zeros(nx)

for i, x_pos in enumerate(x_grid):
    for layer_idx, mat in enumerate(layer_sequence):
        if x_pos <= layer_positions[layer_idx + 1]:
            k_grid[i] = materials[mat]['k']
            rho_grid[i] = materials[mat]['rho']
            cp_grid[i] = materials[mat]['Cp']
            break


T_inner_bc = 37 + 273.15
T_initial = 275 + 273.15

orbit_period = 90 * 60
sunlight_duration = 45 * 60

solar_constant_leo = 1367
earth_ir_flux = 50

sigma = 5.67e-8  # Stefan-Boltzmann constant
epsilon = 0.85   # Emissivity 
alpha_solar = 0.7  # Solar absorptivity

def incident_flux(elapsed_time):
    """ Returns (Solar Flux, Earth IR Flux) based on orbit phase """
    phase = elapsed_time % orbit_period

    if CASE == "extreme_hot":
        return 1500, earth_ir_flux
    elif CASE == "extreme_cold":
        return 0, 0
    if phase < sunlight_duration:
        # Day side: Exposed to both Sun and Earth IR
        return solar_constant_leo, earth_ir_flux
    else:
        # Eclipse: Exposed only to Earth IR
        return 0, earth_ir_flux


def net_heat_flux(T_surface, elapsed_time):
    q_solar, q_ir = incident_flux(elapsed_time)
    
    # Outer surface absorbs solar based on alpha, and IR based on epsilon
    q_in = (alpha_solar * q_solar) + (epsilon * q_ir)
    
    T_space = 3  # Deep space background (K)
    q_out = epsilon * sigma * (T_surface**4 - T_space**4)
    
    return q_in - q_out

total_simulation_time = 2 * orbit_period
dt = 5
num_timesteps = int(total_simulation_time / dt) + 1
time_array = np.linspace(0, total_simulation_time, num_timesteps)

print(f"TIME INTEGRATION SETUP:")
print(f"  Scheme: Crank-Nicolson (Position-Dependent k)")
print(f"  Total simulation time: {total_simulation_time/60:.1f} min")
print(f"  Time step: {dt} s | Steps: {num_timesteps} | Grid points: {nx}")
print(f"  dx = {dx*1000:.4f} mm\n")

A_matrix = np.zeros((nx, nx))
B_matrix = np.zeros((nx, nx))

# Internal Nodes
for i in range(1, nx - 1):
    k_plus = 2 * k_grid[i] * k_grid[i + 1] / (k_grid[i] + k_grid[i + 1])
    k_minus = 2 * k_grid[i] * k_grid[i - 1] / (k_grid[i] + k_grid[i - 1])

    a = dt / (2 * dx**2 * rho_grid[i] * cp_grid[i])

    A_matrix[i, i - 1] = -a * k_minus
    A_matrix[i, i]     = 1 + a * (k_plus + k_minus)
    A_matrix[i, i + 1] = -a * k_plus

    B_matrix[i, i - 1] = a * k_minus
    B_matrix[i, i]     = 1 - a * (k_plus + k_minus)
    B_matrix[i, i + 1] = a * k_plus

# Corrected Outer Boundary Ghost-Node Energy Balance Formulation
r_0 = (k_grid[0] * dt) / (rho_grid[0] * cp_grid[0] * dx**2)
A_matrix[0, 0] = 1 + r_0
A_matrix[0, 1] = -r_0
B_matrix[0, 0] = 1 - r_0
B_matrix[0, 1] = r_0

# Inner Boundary Dirichlet
A_matrix[-1, :] = 0
A_matrix[-1, -1] = 1
B_matrix[-1, :] = 0
B_matrix[-1, -1] = 0

T_current = np.full(nx, T_initial)

T_outer_history, T_inner_history = [], []
T_snapshots, snapshot_times = [], []
Q_history, heat_flux_history = [], []

print("Running transient solver...")

for step in range(num_timesteps):
    current_time = time_array[step]
    Q_ext = net_heat_flux(T_current[0], current_time)

    rhs = B_matrix @ T_current
    
    # Apply flux to outer boundary accurately using Ghost Node balance
    rhs[0] += (2 * dt * Q_ext) / (rho_grid[0] * cp_grid[0] * dx)
    
    # Apply Dirichlet to inner boundary
    rhs[-1] = T_inner_bc

    try:
        T_next = np.linalg.solve(A_matrix, rhs)
    except np.linalg.LinAlgError:
        print(f"WARNING: Singular matrix at step {step}")
        T_next = T_current.copy()
        
    T_next = np.clip(T_next, 50, 2000)
    
    # Fourier Heat Flux at Surface Layer
    q_surface = -k_grid[0] * (T_next[1] - T_next[0]) / dx

    # Logging
    Q_history.append(Q_ext)
    heat_flux_history.append(q_surface)
    T_outer_history.append(T_next[0])
    T_inner_history.append(T_next[-1])

    if step % max(1, (num_timesteps // 10)) == 0:
        T_snapshots.append(T_next.copy())
        snapshot_times.append(current_time / 60)
        
    T_current = T_next

# Data Alignment Fix
minutes = (np.arange(len(T_outer_history)) * dt) / 60
T_outer_C = np.array(T_outer_history) - 273.15
T_inner_C = np.array(T_inner_history) - 273.15
Q_array = np.array(Q_history)

print(f"✓ Completed {num_timesteps} steps.\n")

#Radiation Attenuation Analysis

print("RADIATION ATTENUATION ANALYSIS (Empirical Proxy Model):\n")

configs = {
    'Kevlar only': ['Kevlar'],
    'Nomex only': ['Nomex'],
    'MLI only': ['MLI'],
    'MLI + Kevlar': ['MLI', 'Kevlar'],
    'Nomex + Kevlar': ['Nomex', 'Kevlar'],
    'All 3 layers': ['Kevlar', 'Nomex', 'MLI'],
}

radiation_data = {
    'Kevlar': {'thickness_cm': materials['Kevlar']['thickness'] * 100, 'density_g_cm3': materials['Kevlar']['rho'] / 1000},
    'Nomex':  {'thickness_cm': materials['Nomex']['thickness'] * 100,  'density_g_cm3': materials['Nomex']['rho'] / 1000},
    'MLI':    {'thickness_cm': materials['MLI']['thickness'] * 100,    'density_g_cm3': materials['MLI']['rho'] / 1000}
}

for mat in radiation_data:
    radiation_data[mat]['areal_density'] = radiation_data[mat]['thickness_cm'] * radiation_data[mat]['density_g_cm3']


energy_cases = {
    "Low Energy (X-ray proxy)": 5.0,
    "Medium Energy (~1 MeV proxy)": 0.06,
    "High Energy (Primary GCR proxy)": 0.01
}

def compute_attenuation(layer_list, mu_value):
    total_tau = 0.0
    for layer in layer_list:
        if layer in radiation_data:
            total_tau += mu_value * radiation_data[layer]['areal_density']
    return np.exp(-total_tau)

attenuation_results = {}
for energy_label, mu_val in energy_cases.items():
    attenuation_results[energy_label] = {}
    for config_name, layers in configs.items():
        transmission = compute_attenuation(layers, mu_val)
        attenuation_results[energy_label][config_name] = (1 - transmission) * 100

selected_energy = "Medium Energy (~1 MeV proxy)"

for energy_label in attenuation_results:
    print(f"--- {energy_label} ---")
    for config, atten in attenuation_results[energy_label].items():
        print(f"  {config:20s}: {atten:6.2f}% attenuated")

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Temperature profiles
ax1 = fig.add_subplot(gs[0, 0])
x_mm = x_grid * 1000
colors_temp = plt.cm.coolwarm(np.linspace(0, 1, len(T_snapshots)))
for idx, (T_snap, t_snap) in enumerate(zip(T_snapshots, snapshot_times)):
    ax1.plot(x_mm, T_snap - 273.15, color=colors_temp[idx], label=f't = {t_snap:.0f} min', linewidth=1.5)
for pos in layer_positions[1:-1]:
    ax1.axvline(pos * 1000, color='black', linestyle='--', alpha=0.4)
ax1.set_xlabel('Distance from outer surface (mm)', fontsize=11)
ax1.set_ylabel('Temperature (°C)', fontsize=11)
ax1.set_title('Temperature Profiles Over Time', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Surface temperatures
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(minutes, T_outer_C, label='Outer surface (x=0)', linewidth=2.5, color='red')
ax2.plot(minutes, T_inner_C, label='Inner surface (boundary)', linewidth=2.5, color='blue')
ax2.axhline(37, color='blue', linestyle=':', alpha=0.5)
ax2.set_xlabel('Simulation Time (minutes)', fontsize=11)
ax2.set_ylabel('Temperature (°C)', fontsize=11)
ax2.set_title('Surface Temperature Evolution', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: First Orbit Detail
ax3 = fig.add_subplot(gs[1, 0])
orbit1_idx = len(minutes) // 2
ax3.plot(minutes[:orbit1_idx], T_outer_C[:orbit1_idx], linewidth=2.5, color='darkorange')
ax3.fill_between(minutes[:orbit1_idx], T_outer_C[:orbit1_idx], alpha=0.3, color='darkorange')
ax3.set_xlabel('Time in First Orbit (minutes)', fontsize=11)
ax3.set_ylabel('Temperature (°C)', fontsize=11)
ax3.set_title('First Orbit Detail', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Radiation Attenuation
ax4 = fig.add_subplot(gs[1, 1])
config_names = list(attenuation_results[selected_energy].keys())
attenuation_values = list(attenuation_results[selected_energy].values())
bars = ax4.bar(range(len(config_names)), attenuation_values, color=plt.cm.Spectral(np.linspace(0, 1, len(config_names))), edgecolor='black')
for bar, val in zip(bars, attenuation_values):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax4.set_ylabel('Attenuation (%)', fontsize=11)
ax4.set_title(f'Radiation Attenuation ({selected_energy})', fontsize=12, fontweight='bold')
ax4.set_xticks(range(len(config_names)))
ax4.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
ax4.set_ylim([0, max(attenuation_values) * 1.15])
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Heat Flux
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(minutes, Q_array, linewidth=2.5, color='gold')
ax5.fill_between(minutes, Q_array, alpha=0.3, color='gold')
ax5.set_xlabel('Simulation Time (minutes)', fontsize=11)
ax5.set_ylabel('Net External Heat Flux (W/m²)', fontsize=11)
ax5.set_title('Orbital Heat Flux Profile', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Material Properties
ax6 = fig.add_subplot(gs[2, 1])
mat_names = list(materials.keys())
k_values = [materials[m]['k'] for m in mat_names]
alpha_values = [materials[m]['k'] / (materials[m]['rho'] * materials[m]['Cp']) for m in mat_names]
ax6_2 = ax6.twinx()
ax6.bar([x - 0.2 for x in range(len(mat_names))], k_values, width=0.4, label='Thermal Conductivity', color='steelblue', edgecolor='black')
ax6_2.bar([x + 0.2 for x in range(len(mat_names))], alpha_values, width=0.4, label='Thermal Diffusivity', color='coral', edgecolor='black')
ax6.set_ylabel('Conductivity k (W/m·K)', fontsize=10, color='steelblue')
ax6_2.set_ylabel('Diffusivity α (m²/s)', fontsize=10, color='coral')
ax6.set_xticks(range(len(mat_names)))
ax6.set_xticklabels(mat_names, fontsize=10)
ax6.set_title('Material Properties', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.savefig('space_suit_thermal_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved detailed plots to: space_suit_thermal_analysis.png")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*75)
print("ANALYSIS SUMMARY")
print("="*75)

print("\n1. THERMAL PERFORMANCE:")
print(f"   Max outer surface temp:           {max(T_outer_C):7.2f} °C")
print(f"   Min outer surface temp:           {min(T_outer_C):7.2f} °C")
print(f"   Inner surface temp (fixed):       {T_inner_C[-1]:7.2f} °C")

print("\n2. ORBITAL DYNAMICS:")
print(f"   Orbital period:                   {orbit_period/60:.1f} minutes")
print(f"   Solar constant (LEO):             {solar_constant_leo:.0f} W/m²")
print(f"   Earth IR flux:                    {earth_ir_flux:.0f} W/m²")

best_config = max(attenuation_results[selected_energy], key=attenuation_results[selected_energy].get)
best_value = attenuation_results[selected_energy][best_config]
print(f"\n3. BEST RADIATION CONFIGURATION:     {best_config} ({best_value:.2f}%)")

print("\n4. NUMERICAL SETUP:")
print(f"   Integration scheme:               Crank-Nicolson (2nd Order)")
print(f"   Time step:                        {dt} seconds")
print(f"   Grid spacing:                     {dx*1000:.4f} mm")

print("\n" + "="*75)
print("Analysis complete. Results ready for documentation and deployment.")
print("="*75 + "\n")

plt.show()
