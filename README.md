# Hazardous Facility Relocation Optimization: Gabès Phosphate Case

This project provides a robust decision-support framework for the relocation of the Groupe Chimique Tunisien (GCT) phosphate processing facility in Gabès, Tunisia. Utilizing **Multi-Objective Goal Programming (MOGP)** and **Monte Carlo Simulation**, the study evaluates potential sites based on population safety, economic costs, water security, and environmental integrity under conditions of uncertainty.

## 🎓 Academic Context
- **Institution:** Tunis Business School (TBS), University of Tunis
- **Course:** Business Optimization
- **Authors:** Khouloud Ben Younes & Montaha Ghabri
- **Evaluated by:** Pr. Dr. H. Essid
- **Academic Year:** 2025-2026

## 📝 Project Overview
The Gabès industrial complex is a pillar of the Tunisian economy but faces a severe socio-environmental crisis. This project addresses the "wicked problem" of relocation by:
1.  **Screening:** Filtering 24 candidate sites down to 6 feasible locations using stochastic screening.
2.  **Modeling:** Implementing a Lexicographic Goal Programming model that prioritizes human health (97% reduction in population exposure) over fiscal costs.
3.  **Simulation:** Running 1,000 Monte Carlo iterations per site to account for parameter uncertainty (costs, hydrogeology, and demographics).
4.  **Optimal Solution:** Identifying **Boughrara (Medenine)** as the robust optimal site.

## 📁 Directory Structure
```text
.
├── Data and Plot Generation/      # Core computational scripts and datasets
│   ├── Monte Carlo Simulation.py  # Script for stochastic parameter sampling
│   ├── Plots_Generation.py        # Generates radar charts and deviation plots
│   ├── Rain_Prediction.py         # Supporting meteorological analysis
│   ├── System Parameters.xlsx     # Input data for the optimization model
│   ├── monte_carlo_results.csv    # Exported simulation data
│   └── tun_pop_CN_..._Image.png   # Population density visualization
├── GCT Docs/                      # External source material and audits
│   ├── 20100276_eia_fr.pdf        # Environmental Impact Assessment
│   └── Rapport_Audit_E_S-GCT.pdf  # Technical audit of the Gabès complex
├── Latex Report Files/            # Source files for the final document
│   ├── images/                    # Figure assets (raw deviations, radar charts)
│   ├── sections/*.tex             # Modular LaTeX chapters (Introduction, Methodology, etc.)
│   └── main.tex                   # Main LaTeX compiler file
└── Project Report.pdf             # The final comprehensive research paper
```

## 🛠️ Requirements & Installation
The analysis is implemented in **Python 3.12**. To reproduce the results, you will need the following libraries:

```bash
pip install numpy scipy pandas pulp matplotlib seaborn
```

*   **PuLP:** Used for solving the Mixed-Integer Linear Program (MILP).
*   **SciPy/NumPy:** Used for the Cholesky decomposition and stochastic sampling.
*   **Pandas:** For data manipulation and results aggregation.

## 🚀 Usage
1.  **Run Simulation:** Execute `Monte Carlo Simulation.py` to generate the parameter distributions and perform feasibility screening.
2.  **Optimize:** The optimization logic is embedded within the simulation scripts to find the optimal site (Boughrara) based on median values.
3.  **Visualize:** Run `Plots_Generation.py` to produce the Radar Performance Comparison and Raw Deviation charts found in the report.

## 📊 Key Findings
- **Optimal Site:** Boughrara (Medenine).
- **Safety Impact:** 97% reduction in population exposure (from 8,500 to 280 persons).
- **Economic Trade-off:** Requires a 506 Million TND "safety premium" over the status quo.
- **Robustness:** Boughrara remains the optimal choice in 5 out of 6 weight sensitivity scenarios, proving it is not highly dependent on specific parameter variations (±20%).

## 📜 License
## 📜 License & Contact
This project was prepared for academic purposes at Tunis Business School. For inquiries regarding the data or methodology, please reach out to [moontahaghabry@gmail.com](mailto:moontahaghabry@gmail.com).
