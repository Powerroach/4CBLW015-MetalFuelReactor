#####################################################################################
# Reactor 1 — Dry Cycle Iron-Powder Combustor
# Energy Balance Script
# =============================================
# Course:  4CBLW015 — Designing a metal fuel energy system for campus power
# Group:   2
# Author:  Rowan Cornet
#
# Design basis: Neuron building, TU/e campus
# Target:       509 kW peak heat demand
#####################################################################################


#####################################################################################
# 1: IMPORTS
#####################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq


#####################################################################################
# 2: CONSTANTS AND MOLAR MASSES
#####################################################################################

R         = 8.314        # Universal gas constant [J/(mol·K)]
T_ref     = 298.15       # Reference temperature [K]

M_Fe      = 55.845e-3    # Molar mass Fe    [kg/mol]
M_Fe2O3   = 159.69e-3    # Molar mass Fe2O3 [kg/mol]
M_O2      = 32.0e-3      # Molar mass O2    [kg/mol]
M_N2      = 28.014e-3    # Molar mass N2    [kg/mol]

rho_Fe    = 7874         # Density of iron     [kg/m³]
rho_Fe2O3 = 5240         # Density of hematite [kg/m³]


#####################################################################################
# 3: THERMOCHEMISTRY CHECK
#####################################################################################
# We verify the LHV of iron reported by Bergthorson 2015 (7.36 MJ/kg)
# by calculating it ourselves using Hess's law and NIST formation enthalpies.
#
# Reaction: 4 Fe(s) + 3 O2(g) → 2 Fe2O3(s)
#
# Standard enthalpies of formation at 298.15 K (NIST Webbook):
#   ΔHf° Fe(s)    =    0.0    kJ/mol  (reference element)
#   ΔHf° O2(g)    =    0.0    kJ/mol  (reference element)
#   ΔHf° Fe2O3(s) = -825.5   kJ/mol
#
# ΔH_rxn = 2*(-825.5) - [4*(0) + 3*(0)] = -1651.0 kJ per mol reaction
# Per mole of Fe: ΔH = -1651.0 / 4 = -412.75 kJ/mol Fe
# LHV = |ΔH per mol Fe| / M_Fe = 412750 / 0.055845 = 7.391 MJ/kg

dHf_Fe2O3 = -825.5e3    # [J/mol]  NIST Webbook
dHf_Fe    = 0.0          # [J/mol]  reference element
dHf_O2    = 0.0          # [J/mol]  reference element

# ΔH per mole of Fe (divide total reaction enthalpy by 4 mol Fe)
dH_rxn_per_mol_Fe = (2 * dHf_Fe2O3 - 4 * dHf_Fe - 3 * dHf_O2) / 4   # J/mol Fe

# LHV of iron
LHV_Fe = abs(dH_rxn_per_mol_Fe) / M_Fe    # J/kg Fe

print("=" * 65)
print("  THERMOCHEMISTRY CHECK")
print("=" * 65)
print(f"  ΔH_rxn per mol Fe:              {dH_rxn_per_mol_Fe/1e3:.2f} kJ/mol Fe")
print(f"  LHV of iron (calculated):       {LHV_Fe/1e6:.4f} MJ/kg Fe")
print(f"  LHV of iron (Bergthorson 2015): 7.36 MJ/kg Fe")
print(f"  Difference:                     {abs(LHV_Fe/1e6 - 7.36)/7.36*100:.2f}%  ✓")


#####################################################################################
# 4: HEAT CAPACITY — SHOMATE EQUATION
#####################################################################################
# Cp is not constant — it changes with temperature.
# NIST Webbook gives Cp as the Shomate equation:
#
#   Cp(T) = A + B*t + C*t^2 + D*t^3 + E/t^2     [J/(mol·K)]
#   where t = T[K] / 1000
#
# IMPORTANT: NIST provides TWO coefficient sets per species, each valid
# over a different temperature range. Using the wrong set gives
# scientifically incorrect results. We select the correct set automatically.

# ── O2 coefficients (NIST Webbook) ───────────────────────────────────────────
shomate_O2_low = {               # valid 100 – 700 K
    'A':  31.32234,
    'B': -20.23531,
    'C':  57.86644,
    'D': -36.50624,
    'E':  -0.007374,
    'T_min': 100,
    'T_max': 700,
}
shomate_O2_high = {              # valid 700 – 2000 K
    'A':  30.03235,
    'B':   8.772972,
    'C':  -3.988133,
    'D':   0.788313,
    'E':  -0.741599,
    'T_min': 700,
    'T_max': 2000,
}

# ── N2 coefficients (NIST Webbook) ───────────────────────────────────────────
shomate_N2_low = {               # valid 100 – 700 K
    'A':  28.98641,
    'B':   1.853978,
    'C':  -9.647574,
    'D':  16.63537,
    'E':   0.000117,
    'T_min': 100,
    'T_max': 700,
}
shomate_N2_high = {              # valid 700 – 2000 K
    'A':  26.09200,
    'B':   8.218801,
    'C':  -1.976141,
    'D':   0.159274,
    'E':   0.044434,
    'T_min': 700,
    'T_max': 2000,
}

# ── Fe2O3 coefficients (NIST Webbook) ────────────────────────────────────────
shomate_Fe2O3 = {                # valid 298 – 1800 K  (single range)
    'A':  93.43834,
    'B': 108.3577,
    'C': -50.86003,
    'D':  25.58911,
    'E':  -1.611034,
    'T_min': 298,
    'T_max': 1800,
}


def Cp_shomate(T_K, low_coeff, high_coeff=None):
    """
    Molar heat capacity using NIST Shomate equation.
    Automatically selects the correct coefficient set for the temperature.

    Parameters
    ----------
    T_K        : float   Temperature [K]
    low_coeff  : dict    Shomate coefficients (low T range, or only range)
    high_coeff : dict    Shomate coefficients (high T range), or None

    Returns
    -------
    Cp : float   Molar heat capacity [J/(mol·K)]
    """
    if high_coeff is None:
        coeff = low_coeff
    elif T_K <= low_coeff['T_max']:
        coeff = low_coeff
    else:
        coeff = high_coeff

    t = T_K / 1000.0
    Cp = (coeff['A']
          + coeff['B'] * t
          + coeff['C'] * t**2
          + coeff['D'] * t**3
          + coeff['E'] / t**2)
    return Cp   # J/(mol·K)


def enthalpy_change(T1_K, T2_K, low_coeff, high_coeff=None):
    """
    Enthalpy change ΔH = integral of Cp dT from T1 to T2.

    Handles integration ranges that span two Shomate coefficient ranges
    by splitting the integral at the boundary temperature (700 K for O2/N2).

    Parameters
    ----------
    T1_K       : float   Initial temperature [K]
    T2_K       : float   Final temperature [K]
    low_coeff  : dict    Shomate coefficients (low T range, or only range)
    high_coeff : dict    Shomate coefficients (high T range), or None

    Returns
    -------
    dH : float   Enthalpy change [J/mol]
    """
    # Case 1: only one coefficient set (e.g. Fe2O3)
    if high_coeff is None:
        dH, _ = quad(lambda T: Cp_shomate(T, low_coeff), T1_K, T2_K)
        return dH

    T_boundary = low_coeff['T_max']   # 700 K for O2 and N2

    # Case 2: entire range within low-T coefficients
    if T2_K <= T_boundary:
        dH, _ = quad(lambda T: Cp_shomate(T, low_coeff), T1_K, T2_K)

    # Case 3: entire range within high-T coefficients
    elif T1_K >= T_boundary:
        dH, _ = quad(lambda T: Cp_shomate(T, high_coeff), T1_K, T2_K)

    # Case 4: range spans the boundary — split the integral
    # This applies when integrating from ~293 K up to ~1786 K (adiabatic T)
    else:
        dH_low,  _ = quad(lambda T: Cp_shomate(T, low_coeff),  T1_K,      T_boundary)
        dH_high, _ = quad(lambda T: Cp_shomate(T, high_coeff), T_boundary, T2_K)
        dH = dH_low + dH_high

    return dH   # J/mol


#####################################################################################
# 5: DESIGN BASIS
#####################################################################################

Q_useful    = 509e3      # Target thermal output — Neuron building peak heat [W]
                         # From benchmark estimation: 764 MWh/yr ÷ 1500 hr = 509 kW

lambda_air  = 1.8        # Air excess ratio [-]
                         # Controls flame temperature — 1.5 to 2.0 typical range
                         # λ = 1.8 chosen to keep T_ad safely below Fe3O4 melting point

T_air_in    = 293.15     # Inlet air temperature [K]  (20°C ambient)

T_flue_out  = 423.15     # Flue gas outlet temperature [K]  (150°C)
                         # Chosen to:
                         #   a) Maximise heat recovery into building circuit
                         #   b) Stay above water dew point (~50°C)
                         #   c) Consistent with industrial boiler stack temperatures

f_wall_loss = 0.05       # Wall heat loss fraction [-]
                         # 5% of Q_combustion — standard engineering estimate
                         # for a well-insulated industrial combustion system


#####################################################################################
# 6: STOICHIOMETRY
#####################################################################################
# Reaction: 4 Fe + 3 O2 → 2 Fe2O3
#
# Per mole of Fe burned:
#   O2 consumed (stoich) = 3/4 mol
#   Fe2O3 produced       = 1/2 mol
#   N2 in air            = (3/4)*λ * (79/21) mol  (air is 21% O2, 79% N2 by volume)
#   O2 excess in flue    = (3/4)*(λ-1) mol
#
# Fe2O3 is SOLID — it exits via the cyclone separator, NOT via the flue gas.
# Flue gas = excess O2 + N2 only.

def stoichiometry(m_dot_Fe, lambda_air):
    """
    Calculate all molar and mass flow rates from the iron feed rate.

    Parameters
    ----------
    m_dot_Fe   : float   Iron mass flow rate [kg/s]
    lambda_air : float   Air excess ratio [-]

    Returns
    -------
    dict of molar flow rates [mol/s] and mass flow rates [kg/s]
    """
    n_dot_Fe         = m_dot_Fe / M_Fe                      # mol Fe/s

    # Oxygen flows
    n_dot_O2_stoich  = (3/4) * n_dot_Fe                    # stoichiometric O2 [mol/s]
    n_dot_O2_in      = lambda_air * n_dot_O2_stoich         # actual O2 fed [mol/s]
    n_dot_O2_excess  = n_dot_O2_in - n_dot_O2_stoich       # excess O2 in flue [mol/s]

    # Nitrogen (from air composition: 79/21 N2/O2 molar ratio)
    n_dot_N2         = n_dot_O2_in * (79 / 21)             # N2 [mol/s]

    # Products
    n_dot_Fe2O3      = (1/2) * n_dot_Fe                    # Fe2O3 produced [mol/s]

    # Mass flow rates
    m_dot_O2_in      = n_dot_O2_in   * M_O2                # [kg/s]
    m_dot_N2         = n_dot_N2      * M_N2                 # [kg/s]
    m_dot_air        = m_dot_O2_in   + m_dot_N2             # [kg/s]
    m_dot_Fe2O3      = n_dot_Fe2O3   * M_Fe2O3              # [kg/s]

    # Flue gas: N2 + excess O2 only (Fe2O3 is solid, collected by cyclone)
    m_dot_flue       = m_dot_N2 + n_dot_O2_excess * M_O2   # [kg/s]
    n_dot_flue_O2    = n_dot_O2_excess                      # [mol/s]
    n_dot_flue_N2    = n_dot_N2                             # [mol/s]

    return {
        "n_dot_Fe":        n_dot_Fe,
        "n_dot_O2_stoich": n_dot_O2_stoich,
        "n_dot_O2_in":     n_dot_O2_in,
        "n_dot_O2_excess": n_dot_O2_excess,
        "n_dot_N2":        n_dot_N2,
        "n_dot_Fe2O3":     n_dot_Fe2O3,
        "m_dot_Fe":        m_dot_Fe,
        "m_dot_air":       m_dot_air,
        "m_dot_Fe2O3":     m_dot_Fe2O3,
        "m_dot_flue":      m_dot_flue,
        "n_dot_flue_O2":   n_dot_flue_O2,
        "n_dot_flue_N2":   n_dot_flue_N2,
    }


#####################################################################################
# 7: ENERGY BALANCE — ITERATIVE SOLUTION
#####################################################################################
# The energy balance is:
#   Q_combustion = Q_useful + Q_flue + Q_losses
#
# We cannot solve this directly because m_dot_Fe is unknown.
# We only know Q_useful (our target = 509 kW).
#
# Solution strategy — write a residual function:
#   residual(m_dot_Fe) = Q_useful_calculated - Q_useful_target
#
# When residual = 0, the energy balance is exactly satisfied.
# brentq finds the value of m_dot_Fe that makes residual = 0.
#
# Steps inside the residual:
#   1. Calculate Q_combustion = m_dot_Fe * LHV_Fe
#   2. Calculate Q_losses     = f_wall_loss * Q_combustion
#   3. Calculate Q_flue       = sum(n_dot_i * integral(Cp_i dT, T_in, T_flue))
#   4. Calculate Q_useful_calc = Q_combustion - Q_losses - Q_flue
#   5. Return Q_useful_calc - Q_useful_target

def energy_balance_residual(m_dot_Fe):
    """
    Residual function for the energy balance.
    Returns zero when Q_useful_calculated == Q_useful_target.

    Parameters
    ----------
    m_dot_Fe : float   Iron mass flow rate [kg/s]

    Returns
    -------
    residual : float   [W]
    """
    flows        = stoichiometry(m_dot_Fe, lambda_air)
    Q_combustion = m_dot_Fe * LHV_Fe
    Q_losses     = f_wall_loss * Q_combustion

    # Flue gas enthalpy rise from T_air_in to T_flue_out
    # Uses correct Shomate coefficients for each temperature range
    dH_O2 = enthalpy_change(T_air_in, T_flue_out, shomate_O2_low, shomate_O2_high)
    dH_N2 = enthalpy_change(T_air_in, T_flue_out, shomate_N2_low, shomate_N2_high)

    Q_flue       = (flows["n_dot_flue_O2"] * dH_O2 +
                    flows["n_dot_flue_N2"] * dH_N2)
    Q_useful_calc = Q_combustion - Q_losses - Q_flue

    return Q_useful_calc - Q_useful   # residual [W]


# Solve using Brent's method — finds root in bracket [0.001, 1.0] kg/s
m_dot_Fe_solution = brentq(energy_balance_residual, 0.001, 1.0, xtol=1e-9)

# Recalculate all quantities at the solution
flows        = stoichiometry(m_dot_Fe_solution, lambda_air)
Q_combustion = m_dot_Fe_solution * LHV_Fe
Q_losses     = f_wall_loss * Q_combustion

dH_O2  = enthalpy_change(T_air_in, T_flue_out, shomate_O2_low, shomate_O2_high)
dH_N2  = enthalpy_change(T_air_in, T_flue_out, shomate_N2_low, shomate_N2_high)
Q_flue = (flows["n_dot_flue_O2"] * dH_O2 +
          flows["n_dot_flue_N2"] * dH_N2)

Q_useful_check = Q_combustion - Q_losses - Q_flue
eta_combustor  = Q_useful / Q_combustion


#####################################################################################
# 8: ADIABATIC FLAME TEMPERATURE
#####################################################################################
# Safety check: confirms that the flame temperature at λ = 1.8
# stays below the Fe3O4 melting point of 1597°C (1870 K).
#
# If the flame exceeds this temperature, iron oxide melts into liquid
# droplets that foul the reactor walls and heat exchanger surfaces.
#
# Method: at adiabatic conditions all combustion enthalpy heats the products.
#   q_released = sum(n_i * integral(Cp_i dT, T_in, T_ad))
#
# We solve for T_ad using brentq.

def T_adiabatic(lambda_air, T_in=T_air_in):
    """
    Adiabatic flame temperature by energy balance on products.

    Parameters
    ----------
    lambda_air : float   Air excess ratio [-]
    T_in       : float   Reactant inlet temperature [K]

    Returns
    -------
    T_ad : float   Adiabatic flame temperature [K]
    """
    q = abs(dH_rxn_per_mol_Fe)   # heat released per mol Fe [J/mol]

    # Moles of products per mole of Fe burned (from stoichiometry)
    n_Fe2O3     = 0.5
    n_O2_excess = 0.75 * (lambda_air - 1)
    n_N2        = 0.75 * lambda_air * (79 / 21)

    def residual_Tad(T_ad):
        # Enthalpy absorbed by each product species from T_in to T_ad
        dH_Fe2O3 = enthalpy_change(T_in, T_ad, shomate_Fe2O3)
        dH_O2    = enthalpy_change(T_in, T_ad, shomate_O2_low, shomate_O2_high)
        dH_N2    = enthalpy_change(T_in, T_ad, shomate_N2_low, shomate_N2_high)

        Q_products = (n_Fe2O3     * dH_Fe2O3 +
                      n_O2_excess * dH_O2    +
                      n_N2        * dH_N2)
        return Q_products - q   # = 0 when T_ad is correct

    T_ad = brentq(residual_Tad, T_in + 1, 2200, xtol=0.1)
    return T_ad

T_ad          = T_adiabatic(lambda_air)
T_melt_Fe3O4  = 1870   # K — melting point of Fe3O4


#####################################################################################
# 9: PRINT RESULTS
#####################################################################################

print("\n" + "=" * 65)
print("  ENERGY BALANCE RESULTS — REACTOR 1 (DRY CYCLE COMBUSTOR)")
print("=" * 65)

print(f"\n--- Design Basis ---")
print(f"  Target heat output (Neuron):    {Q_useful/1e3:.1f} kW")
print(f"  Air excess ratio (λ):           {lambda_air:.1f}")
print(f"  Air inlet temperature:          {T_air_in - 273.15:.0f} °C")
print(f"  Flue gas outlet temperature:    {T_flue_out - 273.15:.0f} °C")
print(f"  Wall loss fraction:             {f_wall_loss*100:.0f} %")

print(f"\n--- Stoichiometry & Mass Flows ---")
print(f"  Iron feed rate:                 {m_dot_Fe_solution*3600:.3f} kg/hr  "
      f"({m_dot_Fe_solution*1000:.3f} g/s)")
print(f"  Air flow rate:                  {flows['m_dot_air']*3600:.2f} kg/hr")
print(f"  Fe2O3 produced:                 {flows['m_dot_Fe2O3']*3600:.3f} kg/hr")
print(f"  Flue gas flow rate:             {flows['m_dot_flue']*3600:.2f} kg/hr")
print(f"  Stoich O2 consumed:             {flows['n_dot_O2_stoich']*3600:.4f} mol/hr")
print(f"  Excess O2 in flue:              {flows['n_dot_O2_excess']*3600:.4f} mol/hr")

print(f"\n--- Energy Balance ---")
print(f"  Q_combustion:                   {Q_combustion/1e3:.2f} kW  (iron LHV × ṁ_Fe)")
print(f"  Q_useful (target):              {Q_useful/1e3:.2f} kW  (→ building heating)")
print(f"  Q_flue (flue gas loss):         {Q_flue/1e3:.2f} kW  (N2 + excess O2)")
print(f"  Q_losses (wall loss):           {Q_losses/1e3:.2f} kW  ({f_wall_loss*100:.0f}% of Q_comb)")
print(f"  Balance check (should = 0):     {(Q_useful_check - Q_useful)/1e3:.6f} kW  ✓")

print(f"\n--- Efficiency ---")
print(f"  Combustor efficiency (η):       {eta_combustor*100:.1f} %")
print(f"  Q_useful / Q_combustion:        {Q_useful:.1f} / {Q_combustion:.1f} W")

print(f"\n--- Flame Temperature ---")
print(f"  Adiabatic flame temperature:    {T_ad:.1f} K  ({T_ad - 273.15:.1f} °C)")
print(f"  Fe3O4 melting point:            {T_melt_Fe3O4} K  (1597 °C)")
if T_ad > T_melt_Fe3O4:
    print(f"  ⚠  WARNING: T_ad exceeds Fe3O4 melting point!")
    print(f"     → Increase λ or use flue gas recirculation")
else:
    print(f"  ✓  T_ad safely below Fe3O4 melting point")
    print(f"  Safety margin:                  {T_melt_Fe3O4 - T_ad:.1f} K")

print(f"\n--- Annual Figures (at 1500 full-load hr/yr) ---")
operating_hours = 1500
annual_Fe  = m_dot_Fe_solution * 3600 * operating_hours
annual_ox  = flows['m_dot_Fe2O3'] * 3600 * operating_hours
print(f"  Annual iron consumption:        {annual_Fe:.0f} kg/yr  ({annual_Fe/1000:.1f} t/yr)")
print(f"  Annual Fe2O3 produced:          {annual_ox:.0f} kg/yr  ({annual_ox/1000:.1f} t/yr)")


#####################################################################################
# 10: PLOTTING
#####################################################################################

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Reactor 1 — Dry Cycle Iron-Powder Combustor\nEnergy Balance",
             fontsize=13, fontweight='bold')

# ── Plot 1: Energy balance bar chart ─────────────────────────────────────────
ax1 = axes[0]
categories = ['Q_combustion', 'Q_useful', 'Q_flue', 'Q_losses']
values     = [Q_combustion/1e3, Q_useful/1e3, Q_flue/1e3, Q_losses/1e3]
colours    = ['#333333', '#2E7D32', '#888888', '#AAAAAA']
bars = ax1.bar(categories, values, color=colours, edgecolor='white', linewidth=0.8)
ax1.set_ylabel('Power [kW]')
ax1.set_title('Energy Balance Overview')
ax1.set_xticks(range(len(categories)))
ax1.set_xticklabels(categories, rotation=15, ha='right', fontsize=9)
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f} kW', ha='center', va='bottom', fontsize=8)
ax1.grid(axis='y', alpha=0.3)

# ── Plot 2: Flame temperature vs air excess ratio ─────────────────────────────
ax2 = axes[1]   # FIX: ax2 was missing in original code
lambda_range = np.linspace(1.1, 3.0, 40)
T_ad_range   = []
for lam in lambda_range:
    try:
        T_ad_range.append(T_adiabatic(lam))
    except Exception:
        T_ad_range.append(np.nan)

ax2.plot(lambda_range, np.array(T_ad_range) - 273.15, 'k-', linewidth=2)
ax2.axhline(y=1597, color='gray', linestyle='--',
            label='Fe₃O₄ melting point (1597 °C)')
ax2.axvline(x=lambda_air, color='black', linestyle=':',
            label=f'Design λ = {lambda_air}')
ax2.fill_between(lambda_range,
                 np.array(T_ad_range) - 273.15,
                 1597,
                 where=np.array(T_ad_range) - 273.15 > 1597,
                 alpha=0.15, color='red', label='Liquid oxide region')
ax2.set_xlabel('Air excess ratio λ [-]')
ax2.set_ylabel('Adiabatic flame temperature [°C]')
ax2.set_title('Flame Temperature vs Air Excess Ratio')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_xlim(1.1, 3.0)

# ── Plot 3: Cp vs Temperature for flue gas species ───────────────────────────
ax3 = axes[2]
T_range = np.linspace(300, 2000, 200)
Cp_O2   = [Cp_shomate(T, shomate_O2_low, shomate_O2_high) for T in T_range]
Cp_N2   = [Cp_shomate(T, shomate_N2_low, shomate_N2_high) for T in T_range]

ax3.plot(T_range - 273.15, Cp_O2, 'k-',  linewidth=2, label='O₂ (NIST Shomate)')
ax3.plot(T_range - 273.15, Cp_N2, 'k--', linewidth=2, label='N₂ (NIST Shomate)')
ax3.axvline(x=T_air_in - 273.15,  color='gray',  linestyle=':',
            alpha=0.7, label=f'T_in = {T_air_in - 273.15:.0f} °C')
ax3.axvline(x=T_flue_out - 273.15, color='black', linestyle=':',
            alpha=0.7, label=f'T_flue = {T_flue_out - 273.15:.0f} °C')
ax3.axvline(x=427, color='gray', linestyle='--',
            alpha=0.4, label='Shomate boundary (700 K)')
ax3.set_xlabel('Temperature [°C]')
ax3.set_ylabel('Molar heat capacity Cp [J/(mol·K)]')
ax3.set_title('Cp vs Temperature — Flue Gas Species\n(NIST Shomate, dual-range)')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

plt.tight_layout()



#####################################################################################
# 11: SUMMARY TABLE
#####################################################################################

summary = pd.DataFrame({
    "Parameter": [
        "Target heat output",
        "Iron feed rate",
        "Air flow rate",
        "Fe2O3 production rate",
        "Air excess ratio λ",
        "Q_combustion",
        "Q_useful",
        "Q_flue",
        "Q_losses",
        "Combustor efficiency η",
        "Adiabatic flame temperature",
        "Fe3O4 melting point",
        "Safety margin",
        "Annual iron consumption",
        "Annual Fe2O3 produced",
    ],
    "Value": [
        f"{Q_useful/1e3:.1f} kW",
        f"{m_dot_Fe_solution*3600:.3f} kg/hr",
        f"{flows['m_dot_air']*3600:.2f} kg/hr",
        f"{flows['m_dot_Fe2O3']*3600:.3f} kg/hr",
        f"{lambda_air:.1f}",
        f"{Q_combustion/1e3:.2f} kW",
        f"{Q_useful/1e3:.2f} kW",
        f"{Q_flue/1e3:.2f} kW",
        f"{Q_losses/1e3:.2f} kW",
        f"{eta_combustor*100:.1f} %",
        f"{T_ad - 273.15:.1f} °C",
        f"1597 °C",
        f"{T_melt_Fe3O4 - T_ad:.1f} K",
        f"{annual_Fe:.0f} kg/yr",
        f"{annual_ox:.0f} kg/yr",
    ],
    "Source / Note": [
        "Neuron building benchmark estimate",
        "From energy balance (brentq solve)",
        "From stoichiometry",
        "From stoichiometry",
        "Design choice — controls T_flame",
        "m_dot_Fe × LHV_Fe",
        "Design target",
        "N2 + excess O2 sensible heat (NIST Shomate, dual-range)",
        f"{f_wall_loss*100:.0f}% of Q_combustion",
        "Q_useful / Q_combustion",
        "NIST Shomate (dual-range), brentq solve",
        "Fe3O4 phase diagram",
        "T_melt - T_ad",
        "At 1500 full-load hr/yr",
        "At 1500 full-load hr/yr",
    ]
})

print("\n" + "=" * 65)
print("  SUMMARY TABLE")
print("=" * 65)
print(summary.to_string(index=False))


plt.show()
