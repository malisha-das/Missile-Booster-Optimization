import streamlit as st
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

rocket_data = {}
# Constants
g = 9.81
Isp = 248
Ve = Isp * g

# Geometry
OD = 0.75
ID = 0.6
Thickness = OD - ID
Total_length = 7.5
Bulkhead_length = 0.3
Bulkhead_density = 2700

# Material Densities
rho_structure = 2700
rho_fuel = 985

# Fixed values
payload_mass = 250
T = 10  # target delta-v stage 1 contribution
required_mass_ratio_1st_stage = np.exp(T / (2 * Isp))

st.title("3-Stage Rocket Optimizer")

# User input section
with st.sidebar:
    st.header("Customization")
    custom = st.checkbox("Enter custom dimensions and materials")

    if custom:
        ID = st.number_input("Inner Diameter (m)", value=ID)
        rho_structure = st.number_input("Structure Density (kg/m³)", value=rho_structure)
        rho_fuel = st.number_input("Propellant Density (kg/m³)", value=rho_fuel)
        Isp = st.number_input("Specific Impulse (s)", value=Isp)
        Bulkhead_density = st.number_input("Bulkhead Density (kg/m³)", value=Bulkhead_density)
        Ve = Isp * g
        Thickness = OD - ID

# Volume-based mass functions
def structure_mass(L): return rho_structure * ((Thickness**2) * np.pi / 4) * L
def fuel_mass(L): return rho_fuel * ((ID**2) * np.pi / 4) * L

# Bulkhead
bulkhead_volume = (ID**2) * np.pi / 4 * Bulkhead_length
bulkhead_mass = Bulkhead_density * bulkhead_volume

# Objective function
def objective(lengths):
    L1, L2, L3 = lengths
    M_s = list(map(structure_mass, [L1, L2, L3]))
    M_f = list(map(fuel_mass, [L1, L2, L3]))

    M0_1 = sum(M_s + M_f) + 3 * bulkhead_mass + payload_mass
    Mf_1 = M0_1 - M_f[0]
    M0_2 = sum(M_s[1:] + M_f[1:]) + 2 * bulkhead_mass + payload_mass
    Mf_2 = M0_2 - M_f[1]
    M0_3 = M_s[2] + M_f[2] + bulkhead_mass + payload_mass
    Mf_3 = M0_3 - M_f[2]

    return -(Ve * (np.log(M0_1 / Mf_1) + np.log(M0_2 / Mf_2) + np.log(M0_3 / Mf_3)))

def total_length_constraint(L):
    return sum(L) - (Total_length - 3 * Bulkhead_length)

def equal_mass_ratio_constraint(L):
    L1, L2, L3 = L
    M0_1 = sum(structure_mass(l) + fuel_mass(l) for l in L) + 3 * bulkhead_mass + payload_mass
    Mf_1 = M0_1 - fuel_mass(L1)
    MR1 = M0_1 / Mf_1

    M0_2 = structure_mass(L2) + structure_mass(L3) + fuel_mass(L2) + fuel_mass(L3) + 2 * bulkhead_mass + payload_mass
    Mf_2 = M0_2 - fuel_mass(L2)
    MR2 = M0_2 / Mf_2

    M0_3 = structure_mass(L3) + fuel_mass(L3) + bulkhead_mass + payload_mass
    Mf_3 = M0_3 - fuel_mass(L3)
    MR3 = M0_3 / Mf_3

    return [MR1 - MR2, MR2 - MR3]

def stage1_ratio_constraint(L):
    L1, L2, L3 = L
    M0_1 = sum(structure_mass(l) + fuel_mass(l) for l in (L1, L2, L3)) + 3 * bulkhead_mass + payload_mass
    Mf_1 = M0_1 - fuel_mass(L1)
    return (M0_1 / Mf_1) - required_mass_ratio_1st_stage

def stage2_3_equal_ratio_constraint(L):
    L2, L3 = L[1], L[2]
    M0_2 = structure_mass(L2) + structure_mass(L3) + fuel_mass(L2) + fuel_mass(L3) + 2 * bulkhead_mass + payload_mass
    Mf_2 = M0_2 - fuel_mass(L2)
    M0_3 = structure_mass(L3) + fuel_mass(L3) + bulkhead_mass + payload_mass
    Mf_3 = M0_3 - fuel_mass(L3)
    return (M0_2 / Mf_2) - (M0_3 / Mf_3)

def run_optimizer(label, constraints):
    initial_guess = [2.5, 2.5, 2.5]
    result = opt.minimize(objective, initial_guess, constraints=constraints, bounds=[(0, Total_length)] * 3)

    if not result.success:
        st.error(f"{label}: Optimization failed - {result.message}")
        return

    L1, L2, L3 = result.x
    M_s = list(map(structure_mass, [L1, L2, L3]))
    M_f = list(map(fuel_mass, [L1, L2, L3]))

    M0_1 = sum(M_s + M_f) + 3 * bulkhead_mass + payload_mass
    Mf_1 = M0_1 - M_f[0]
    DeltaV1 = Ve * np.log(M0_1 / Mf_1)

    M0_2 = sum(M_s[1:] + M_f[1:]) + 2 * bulkhead_mass + payload_mass
    Mf_2 = M0_2 - M_f[1]
    DeltaV2 = Ve * np.log(M0_2 / Mf_2)

    M0_3 = M_s[2] + M_f[2] + bulkhead_mass + payload_mass
    Mf_3 = M0_3 - M_f[2]
    DeltaV3 = Ve * np.log(M0_3 / Mf_3)

    st.subheader(label)
    st.write(f"Optimized Lengths: L1 = {L1:.4f} m, L2 = {L2:.4f} m, L3 = {L3:.4f} m")
    st.write(f"ΔV Stage 1: {DeltaV1:.2f} m/s, Stage 2: {DeltaV2:.2f} m/s, Stage 3: {DeltaV3:.2f} m/s")
    st.write(f"Total ΔV: {DeltaV1 + DeltaV2 + DeltaV3:.2f} m/s")
    st.write(f"Mass Ratios: MR1 = {M0_1 / Mf_1:.4f}, MR2 = {M0_2 / Mf_2:.4f}, MR3 = {M0_3 / Mf_3:.4f}")
    st.write(f"Fuel Masses: F1 = {M_f[0]:.2f}, F2 = {M_f[1]:.2f}, F3 = {M_f[2]:.2f}")
    st.write(f"Structure Masses: S1 = {M_s[0]:.2f}, S2 = {M_s[1]:.2f}, S3 = {M_s[2]:.2f}")
    rocket_data[label] = {
    'lengths': [L1, L2, L3],
    'dvs': [DeltaV1, DeltaV2, DeltaV3]
}
#graph
def plot_rocket_comparison(rocket_data):
    labels = list(rocket_data.keys())
    stages = ['Stage 1', 'Stage 2', 'Stage 3']
    x = np.arange(len(stages))
    bar_width = 0.35

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    for i, label in enumerate(labels):
        offset = (i - len(labels) / 2) * bar_width + bar_width / 2
        axs[0].bar(x + offset, rocket_data[label]['lengths'], width=bar_width, label=label)
    axs[0].set_title('Optimized Stage Lengths')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(stages)
    axs[0].set_ylabel('Length (m)')
    axs[0].legend()
    axs[0].grid(True)

    for i, label in enumerate(labels):
        offset = (i - len(labels) / 2) * bar_width + bar_width / 2
        axs[1].bar(x + offset, rocket_data[label]['dvs'], width=bar_width, label=label)
    axs[1].set_title('Delta-V Contributions per Stage')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(stages)
    axs[1].set_ylabel('Delta-V (m/s)')
    axs[1].legend()
    axs[1].grid(True)

    dv1 = sum(rocket_data["Rocket 1 (Equal Mass Ratios Across All Stages)"]['dvs'])
    dv2 = sum(rocket_data["Rocket 2 (With MR1 Fixed and MR2 = MR3)"]['dvs'])
    percent_loss = 100 * (dv1 - dv2) / dv1
    axs[1].text(0.9, 0.9, f"ΔV Loss: {percent_loss:.2f}%", transform=axs[1].transAxes,
                fontsize=12, color='red', ha='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round'))

    st.pyplot(fig)

if st.button("Run Optimizations"):
    run_optimizer("Rocket 1 (Equal Mass Ratios Across All Stages)", [
        {'type': 'eq', 'fun': total_length_constraint},
        {'type': 'eq', 'fun': equal_mass_ratio_constraint}
    ])

    run_optimizer("Rocket 2 (With MR1 Fixed and MR2 = MR3)", [
        {'type': 'eq', 'fun': total_length_constraint},
        {'type': 'eq', 'fun': stage1_ratio_constraint},
        {'type': 'eq', 'fun': stage2_3_equal_ratio_constraint}
    ])
    if rocket_data:
        plot_rocket_comparison(rocket_data)

