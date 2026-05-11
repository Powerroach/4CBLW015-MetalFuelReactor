# 4CBLW015-MetalFuelReactor
Iron Power Cycle — Reactor 1 dry cycle oxidation reactor energy balance. CBL project 4CBLW015, TU/e Group 2.


---

## Project Overview
This repository contains the engineering design work for the 
Iron Power Cycle campus energy system, developed as part of the 
Challenge Based Learning course 4CBLW015 at TU/e.

The Iron Power Cycle is a carbon-free energy storage and generation 
system that uses iron powder as a recyclable fuel. Iron is oxidised 
in air to release heat, and the resulting iron oxide is reduced back 
to iron using green hydrogen, closing the cycle with no CO₂ emissions.

Our group is designing a system to supply heat to the Neuron building 
on the TU/e campus (509 kW peak heat demand), using:
- **Reactor 1:** Dry cycle iron-powder combustor (oxidation)
- **Reactor 2:** Fluidised-bed H₂ reduction reactor (regeneration)

---

## Repository Structure
4CBLW015-MetalFuelReactor/
│
├── reactor1_energy_balance.py   # Energy balance script for reactor 1
├── results/
│   ├── reactor1_energy_balance.png   # Output plots
│   └── results_summary.md            # Key numerical results

---

## Reactor 1 — Dry Cycle Iron Powder Combustor
The oxidation reactor burns iron powder directly in air:

**4 Fe + 3 O₂ → 2 Fe₂O₃ + heat**

The script calculates:
- Iron and air feed rates from the building heat demand
- Full energy balance (Q combustion, Q useful, Q flue, Q losses)
- Adiabatic flame temperature using NIST Shomate equation
- Combustor efficiency
- Annual iron and iron oxide mass flows

**Key results:**

| Parameter | Value |
|---|---|
| Iron feed rate | 276 kg/hr |
| Combustor efficiency | 89.8 % |
| Adiabatic flame temperature | 1512.6 °C |
| Safety margin below Fe₃O₄ melting | 84.2 K ✓ |

---

## How to Run
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/4CBLW015-MetalFuelReactor.git

# Install dependencies
pip install numpy scipy matplotlib pandas

# Run the energy balance
python reactor1_energy_balance.py
```

---

## Dependencies
- Python 3.x
- numpy
- scipy
- matplotlib
- pandas

---

## References
1. Bergthorson et al. (2015). *Applied Energy*, 160, 368–382.
2. Neumann et al. (2024). *Applied Energy*, 368, 123476.
3. NIST Webbook — thermodynamic data for Fe, O₂, N₂, Fe₂O₃.
4. van Rooij, N. (2023). PhD dissertation, TU/e.

---

## Authors
Group 2 — 4CBLW015 | Eindhoven University of Technology | 2026
