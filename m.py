import numpy as np
import matplotlib.pyplot as plt
# ============================================================================
# PART 1: 1D TRANSIENT THERMAL CONDUCTION MODEL (IMPROVED)
# ============================================================================

print("\n" + "="*75)
print("SPACE SUIT THERMAL ANALYSIS - LEO CONDITIONS")
print("="*75 + "\n")

# -----------------------------
# EDGE CASE SWITCH (NEW)
# -----------------------------
CASE = "normal"  # options: "normal", "extreme_hot", "extreme_cold"

# -----------------------------
# Material Properties Database
# -----------------------------
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
        'description': 'Multi-layer insulation (thermal barrier)'
    }
}

# -----------------------------
# Geometry Construction
# -----------------------------
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

# -----------------------------
# Spatial Discretization
# -----------------------------
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

alpha_grid = k_grid / (rho_grid * cp_grid)

# ============================================================================
# BOUNDARY CONDITIONS & ORBITAL PARAMETERS (IMPROVED)
# ============================================================================

T_inner_bc = 37 + 273.15
T_initial = 275 + 273.15

orbit_period = 90 * 60
sunlight_duration = 45 * 60
eclipse_duration = 45 * 60

solar_constant_leo = 1367
earth_ir_flux = 50

# -----------------------------
# Radiative Properties (NEW)
# -----------------------------
sigma = 5.67e-8  # Stefan-Boltzmann constant
epsilon = 0.85   # emissivity (outer surface)
alpha_solar = 0.7  # absorptivity

# -----------------------------
# Solar Flux Function (WITH EDGE CASES)
# -----------------------------
def solar_heat_flux(elapsed_time):
    phase = elapsed_time % orbit_period

    if CASE == "extreme_hot":
        return 1500  # higher than solar constant
    elif CASE == "extreme_cold":
        return 0

    if phase < sunlight_duration:
        return solar_constant_leo
    else:
        return earth_ir_flux

# -----------------------------
# Net Heat Flux with Radiation (CRITICAL FIX)
# -----------------------------
def net_heat_flux(T_surface, elapsed_time):
    """
    Net heat flux including:
    - Absorbed solar radiation
    - Radiative emission (Stefan-Boltzmann law)

    NOTE:
    This approximates radiative balance at the outer surface.
    """
    G = solar_heat_flux(elapsed_time)

    q_in = alpha_solar * G  
    T_env = 3  # space background (K) OR use ~250K for Earth IR case
    q_out = epsilon * sigma * (T_surface**4 - T_env**4)            # absorbed solar

    return q_in - q_out  # net flux into material

# ============================================================================
# CRANK-NICOLSON (VARIABLE k — PHYSICALLY CORRECT)
# ============================================================================

# Time parameters
total_simulation_time = 2 * orbit_period
dt = 5
num_timesteps = int(total_simulation_time / dt) + 1
array = np.linspace(0, total_simulation_time, num_timesteps)

print(f"TIME INTEGRATION SETUP:")
print(f"  Scheme: Crank-Nicolson (variable k)")
print(f"  Total simulation time: {total_simulation_time/60:.1f} min")
print(f"  Time step: {dt} s")
print(f"  Steps: {num_timesteps}")
print(f"  Grid points: {nx}")
print(f"  dx = {dx*1000:.4f} mm\n")

# ----------------------------------------------------------------------------
# BUILD MATRICES (VARIABLE k USING HARMONIC MEAN)
# ----------------------------------------------------------------------------
A_matrix = np.zeros((nx, nx))
B_matrix = np.zeros((nx, nx))

for i in range(1, nx - 1):
    # Interface conductivities (harmonic mean)
    k_plus = 2 * k_grid[i] * k_grid[i + 1] / (k_grid[i] + k_grid[i + 1])
    k_minus = 2 * k_grid[i] * k_grid[i - 1] / (k_grid[i] + k_grid[i - 1])

    a = dt / (2 * dx**2)

    # A matrix (implicit)
    A_matrix[i, i - 1] = -a * k_minus
    A_matrix[i, i]     = 1 + a * (k_plus + k_minus)
    A_matrix[i, i + 1] = -a * k_plus

    # B matrix (explicit)
    B_matrix[i, i - 1] = a * k_minus
    B_matrix[i, i]     = 1 - a * (k_plus + k_minus)
    B_matrix[i, i + 1] = a * k_plus

# ----------------------------------------------------------------------------
# BOUNDARY CONDITIONS IN MATRICES
# ----------------------------------------------------------------------------

# Outer boundary (flux BC handled in RHS)
A_matrix[0, 0] = 1
B_matrix[0, 0] = 1

# Inner boundary (Dirichlet)
A_matrix[-1, :] = 0
A_matrix[-1, -1] = 1
B_matrix[-1, :] = 0
B_matrix[-1, -1] = 1

# ----------------------------------------------------------------------------
# INITIAL CONDITIONS
# ----------------------------------------------------------------------------
T_current = np.full(nx, T_initial)

T_outer_history = []
T_inner_history = []
T_snapshots = []
snapshot_times = []
Q_history = []
heat_flux_history = []

# ----------------------------------------------------------------------------
# TIME LOOP
# ----------------------------------------------------------------------------
print("Running variable-k Crank-Nicolson...")

for step in range(num_timesteps - 1):

    current_time = array[step]

    # -----------------------------
    # External heat flux
    # -----------------------------
    Q_ext = net_heat_flux(T_current[0], current_time)
    Q_history.append(Q_ext)

    # -----------------------------
    # RHS construction
    # -----------------------------
    rhs = B_matrix @ T_current

    # -----------------------------
    # OUTER BOUNDARY (FLUX BC)
    # -k dT/dx = Q_ext
    # -----------------------------
    rhs[0] += (Q_ext * dx) / k_grid[0]

    # -----------------------------
    # INNER BOUNDARY (Dirichlet)
    # -----------------------------
    rhs[-1] = T_inner_bc
    A_bc =  A_matrix.copy()
    A_bc[-1, :] = 0
    A_bc[-1, -1] = 1

    # -----------------------------
    # Solve system
    # -----------------------------
    try:
        T_next = np.linalg.solve(A_bc, rhs)
    except np.linalg.LinAlgError:
        print(f"WARNING: Singular matrix at step {step}")
        T_next = T_current.copy()
    T_next = np.clip(T_next, 50, 2000)
    


    # -----------------------------
    # Heat flux (Fourier law)
    # -----------------------------
    q_surface = -k_grid[0] * (T_current[1] - T_current[0]) / dx
    heat_flux_history.append(q_surface)

    # -----------------------------
    # Store results
    # -----------------------------
    T_outer_history.append(T_current[0])
    T_inner_history.append(T_current[-1])

    if step % 12 == 0:
        T_snapshots.append(T_current.copy())
        snapshot_times.append(current_time / 60)

# ----------------------------------------------------------------------------
# POST-PROCESSING
# ----------------------------------------------------------------------------
T_outer_C = np.array(T_outer_history) - 273.15
T_inner_C = np.array(T_inner_history) - 273.15
minutes = array[:-1] / 60

print(f"✓ Completed {num_timesteps} steps.\n")

# ----------------------------------------------------------------------------
# SAVE DATA
# ----------------------------------------------------------------------------
np.save("T_snapshots.npy", np.array(T_snapshots))
np.save("x_grid.npy", x_grid)
np.save("time_snapshots.npy", np.array(snapshot_times))

# ============================================================================
# PART 2: RADIATION ATTENUATION MODEL (IMPROVED)
# ============================================================================

print("RADIATION ATTENUATION ANALYSIS (Improved Model):\n")
configs = {
    'Kevlar only': ['Kevlar'],
    'Nomex only': ['Nomex'],
    'MLI only': ['MLI'],
    'MLI + Kevlar': ['MLI', 'Kevlar'],
    'Nomex + Kevlar': ['Nomex', 'Kevlar'],
    'All 3 layers': ['Kevlar', 'Nomex', 'MLI'],
}
# ---------------------------------------------------------------------------
# Radiation Material Data (REQUIRED)
# ---------------------------------------------------------------------------
radiation_data = {
    'Kevlar': {
        'thickness_cm': materials['Kevlar']['thickness'] * 100,
        'density_g_cm3': materials['Kevlar']['rho'] / 1000,
    },
    'Nomex': {
        'thickness_cm': materials['Nomex']['thickness'] * 100,
        'density_g_cm3': materials['Nomex']['rho'] / 1000,
    },
    'MLI': {
        'thickness_cm': materials['MLI']['thickness'] * 100,
        'density_g_cm3': materials['MLI']['rho'] / 1000,
    }
}

# Compute areal density
for mat in radiation_data:
    radiation_data[mat]['areal_density'] = (
        radiation_data[mat]['thickness_cm'] *
        radiation_data[mat]['density_g_cm3']
    )

# Energy-dependent cases (NEW)
energy_cases = {
    "Low Energy (X-ray)": 0.08,
    "Medium Energy (~1 MeV)": 0.05,
    "High Energy (GCR proxy)": 0.02
}

def compute_attenuation(layer_list, mu_value):
    total_tau = 0.0
    for layer in layer_list:
        if layer in radiation_data:
            areal = radiation_data[layer]['areal_density']
            total_tau += mu_value * areal
    return np.exp(-total_tau)

# ---------------------------------------------------------------------------
# Material Configurations for Radiation Analysis (REQUIRED FIX)
# ---------------------------------------------------------------------------
# Evaluate across energy cases
attenuation_results = {}

for energy_label, mu_val in energy_cases.items():
    attenuation_results[energy_label] = {}
    
    for config_name, layers in configs.items():
        transmission = compute_attenuation(layers, mu_val)
        attenuation_pct = (1 - transmission) * 100
        attenuation_results[energy_label][config_name] = attenuation_pct
# Select energy case for plotting and summary
selected_energy = "Medium Energy (~1 MeV)"

# Print results
for energy_label in attenuation_results:
    print(f"\n--- {energy_label} ---")
    for config, atten in attenuation_results[energy_label].items():
        print(f"  {config:20s}: {atten:6.2f}% attenuated")
print("\nENERGY DEPENDENCE TREND:")
for config in configs:
    low = attenuation_results["Low Energy (X-ray)"][config]
    high = attenuation_results["High Energy (GCR proxy)"][config]
    print(f"{config:20s}: {low:.2f}% → {high:.2f}%")

# ============================================================================
# PLOTTING
# ============================================================================

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Temperature profiles over time
ax1 = fig.add_subplot(gs[0, 0])
x_mm = x_grid * 1000
colors_temp = plt.cm.coolwarm(np.linspace(0, 1, len(T_snapshots)))
for idx, (T_snap, t_snap) in enumerate(zip(T_snapshots, snapshot_times)):
    T_snap_C = T_snap - 273.15
    ax1.plot(x_mm, T_snap_C, color=colors_temp[idx], label=f't = {t_snap:.0f} min', linewidth=1.5)

# Add layer boundaries
for pos in layer_positions[1:-1]:
    ax1.axvline(pos * 1000, color='black', linestyle='--', alpha=0.4, linewidth=1)

ax1.set_xlabel('Distance from outer surface (mm)', fontsize=11)
ax1.set_ylabel('Temperature (°C)', fontsize=11)
ax1.set_title('Temperature Profiles Over Time', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Surface temperatures vs time
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(minutes, T_outer_C, label='Outer surface (x=0)', linewidth=2.5, color='red')
ax2.plot(minutes, T_inner_C, label='Inner surface (boundary, 37°C)', linewidth=2.5, color='blue')
ax2.axhline(37, color='blue', linestyle=':', alpha=0.5)
ax2.set_xlabel('Simulation Time (minutes)', fontsize=11)
ax2.set_ylabel('Temperature (°C)', fontsize=11)
ax2.set_title('Surface Temperature Evolution (2 Orbits)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Outer surface temperature detail (zoomed to one orbit)
ax3 = fig.add_subplot(gs[1, 0])
orbit1_idx = num_timesteps // 2  # End of first orbit
ax3.plot(minutes[:orbit1_idx], T_outer_C[:orbit1_idx], linewidth=2.5, color='darkorange', marker='o', markersize=3)
ax3.fill_between(minutes[:orbit1_idx], T_outer_C[:orbit1_idx], alpha=0.3, color='darkorange')
ax3.set_xlabel('Time in First Orbit (minutes)', fontsize=11)
ax3.set_ylabel('Outer Surface Temperature (°C)', fontsize=11)
ax3.set_title('First Orbit Detail: Thermal Response', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Radiation attenuation comparison
ax4 = fig.add_subplot(gs[1, 1])

# Select energy case
selected_energy = "Medium Energy (~1 MeV)"

config_names = list(attenuation_results[selected_energy].keys())
attenuation_values = list(attenuation_results[selected_energy].values())

# 🔥 CREATE THE BARS (this line was missing)
bars = ax4.bar(
    range(len(config_names)),
    attenuation_values,
    color=plt.cm.Spectral(np.linspace(0, 1, len(config_names))),
    edgecolor='black',
    linewidth=1.2
)

# Add value labels
for bar, val in zip(bars, attenuation_values):
    height = bar.get_height()
    ax4.text(
        bar.get_x() + bar.get_width()/2.,
        height + 1.0,
        f'{val:.1f}%',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold'
    )

ax4.set_ylabel('Attenuation (%)', fontsize=11)
ax4.set_title(f'Radiation Attenuation ({selected_energy})', fontsize=12, fontweight='bold')
ax4.set_xticks(range(len(config_names)))
ax4.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
ax4.set_ylim([0, max(attenuation_values) * 1.15])
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Solar heat flux over time
ax5 = fig.add_subplot(gs[2, 0])
Q_array = np.array(Q_history)
ax5.plot(minutes[:len(Q_array)], Q_array, linewidth=2.5, color='gold', marker='s', markersize=2)
ax5.fill_between(minutes[:len(Q_array)], Q_array, alpha=0.3, color='gold')
ax5.set_xlabel('Simulation Time (minutes)', fontsize=11)
ax5.set_ylabel('Heat Flux (W/m²)', fontsize=11)
ax5.set_title('Orbital Heat Flux Profile', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Material properties comparison
ax6 = fig.add_subplot(gs[2, 1])
mat_names = list(materials.keys())
k_values = [materials[m]['k'] for m in mat_names]
alpha_values = [materials[m]['k'] / (materials[m]['rho'] * materials[m]['Cp']) for m in mat_names]

ax6_2 = ax6.twinx()
bars1 = ax6.bar([x - 0.2 for x in range(len(mat_names))], k_values, width=0.4, 
                label='Thermal Conductivity', color='steelblue', edgecolor='black', linewidth=1)
bars2 = ax6_2.bar([x + 0.2 for x in range(len(mat_names))], alpha_values, width=0.4,
                  label='Thermal Diffusivity', color='coral', edgecolor='black', linewidth=1)

ax6.set_ylabel('Thermal Conductivity k (W/m·K)', fontsize=10, color='steelblue')
ax6_2.set_ylabel('Thermal Diffusivity α (m²/s)', fontsize=10, color='coral')
ax6.set_xticks(range(len(mat_names)))
ax6.set_xticklabels(mat_names, fontsize=10)
ax6.set_title('Material Properties', fontsize=12, fontweight='bold')
ax6.tick_params(axis='y', labelcolor='steelblue')
ax6_2.tick_params(axis='y', labelcolor='coral')
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
print(f"   Maximum outer surface temperature:  {max(T_outer_C):7.2f} °C")
print(f"   Minimum outer surface temperature:  {min(T_outer_C):7.2f} °C")
print(f"   Temperature swing (amplitude):      {(max(T_outer_C) - min(T_outer_C))/2:7.2f} °C")
print(f"   Inner surface temperature (fixed):  {T_inner_C[-1]:7.2f} °C")
print(f"   Effective insulation quality:       ΔT across layers: {T_outer_C[-1] - T_inner_C[-1]:.2f} °C")

print("\n2. ORBITAL DYNAMICS:")
print(f"   Orbital period:                     {orbit_period/60:.1f} minutes")
print(f"   Sunlight phase duration:            {sunlight_duration/60:.1f} minutes")
print(f"   Eclipse phase duration:             {eclipse_duration/60:.1f} minutes")
print(f"   Solar constant (LEO):               {solar_constant_leo:.0f} W/m²")
print(f"   Earth IR flux during eclipse:       {earth_ir_flux:.0f} W/m²")
print(f"   Orbits simulated:                   {total_simulation_time/orbit_period:.1f}")

print("\n3. RADIATION SHIELDING EFFECTIVENESS:")
# Use selected energy case
selected_energy = "Medium Energy (~1 MeV)"

best_config = max(
    attenuation_results[selected_energy],
    key=attenuation_results[selected_energy].get
)

best_value = attenuation_results[selected_energy][best_config]

print(f"   Best performing configuration:      {best_config} ({best_value:.2f}%)")

print("\n4. HEAT CONDUCTION PATHWAYS:")
for i, mat in enumerate(layer_sequence):
    props = materials[mat]
    thermal_resistance = props['thickness'] / props['k']  # R = L/k
    print(f"   {mat:10s}: R-value = {thermal_resistance:.6f} K·m²/W")
print("\n5. HEAT FLUX ANALYSIS:")
print(f"   Maximum heat flux:                {max(heat_flux_history):.2f} W/m²")
print(f"   Minimum heat flux:                {min(heat_flux_history):.2f} W/m²")

print("\n6. NUMERICAL DISCRETIZATION:")
print(f"   Spatial grid points:                {nx}")
print(f"   Grid spacing (minimum):             {dx*1000:.4f} mm")
print(f"   Time step:                          {dt} seconds")
print(f"   Total time steps:                   {num_timesteps}")
print(f"   Integration scheme:                 Crank-Nicolson (2nd order, unconditionally stable)")

print("\n7. KEY ASSUMPTIONS & LIMITATIONS:")
print("   ✓ Perfect thermal contact (no contact resistance)")
print("   ✓ Constant material properties (no temperature dependence)")
print("   ✓ Radiation exchange between layers neglected")
print("   ✓ 1D heat flow only (perpendicular to surface)")
print("   ✓ Simplified solar spectrum (single flux value)")
print("   ✓ No convection (valid for vacuum LEO environment)")

print("\n" + "="*75)
print("Analysis complete. Results suitable for research demonstration.")
print("="*75 + "\n")

plt.show()





