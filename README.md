# Near-Wellbore Salt Precipitation and Preventive Squeeze Strategies in CO₂ Injection Wells

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Code and synthetic dataset accompanying the manuscript:

> **Kaslivtsev, R. (2026).** *Near-Wellbore Salt Precipitation and Geochemical Scaling During CO₂ Injection into Depleted Reservoirs: Mechanisms, Surrogate Predictive Modeling, and Preventive Squeeze Strategies.*

## Overview

Dry supercritical CO₂ ($sc\text{CO}_2$) injection into depleted hydrocarbon reservoirs and deep saline aquifers induces evaporative brine dry-out, capillary backflow, halite ($\text{NaCl}$) precipitation, and secondary carbonate/sulfate scaling ($\text{CaCO}_3$, $\text{CaSO}_4$). This repository provides a reproducible Python workflow that:

1. Generates a **2,500-scenario physics-informed dataset** via Latin Hypercube Sampling (LHS), coupling:
   - Near-wellbore moisture balance and counter-current capillary brine backflow;
   - A modified **Verma–Pruess** permeability impairment relation;
   - A temperature-dependent **Freundlich** adsorption–desorption isotherm for dual-function preventive squeeze treatments.
2. Benchmarks **Multiple Linear Regression**, **Random Forest**, and **XGBoost** (with automatic CUDA GPU detection and CPU fallback) using 5-fold cross-validation and an 80/20 train–test split, including log-target scaling ($\ln(1 + V_{\text{prot}})$) for right-skewed treatment lifetimes.
3. Computes exact **SHAP (TreeExplainer)** attributions and exports all publication figures at 300 DPI.

## Repository Structure

```text
.
├── run_experiment.py                     # End-to-end physics generator, ML benchmark, SHAP & plotting
├── requirements.txt                      # Python dependencies
├── LICENSE                               # MIT License
├── README.md                             # Project documentation
└── results/
    ├── synthetic_ccs_squeeze_dataset.csv # Generated 2,500-scenario dataset
    ├── model_metrics_table.csv           # 5-fold CV and blind test metrics
    ├── Fig1_Parity_Plots.png             # Parity plots (Δk_imp and V_prot)
    ├── Fig2_SHAP_Summary.png             # Global SHAP beeswarm summaries
    ├── Fig3_SHAP_Dependence.png          # Coupled physical SHAP dependence plots
    └── Fig4_Squeeze_Optimization_Map.png # Engineering design maps for squeeze optimization
```

## Quick Start

### 1. Clone the repository and install dependencies

```bash
git clone [https://github.com/rkaslivtsev/ccs-squeeze-surrogate-ml.git](https://github.com/rkaslivtsev/ccs-squeeze-surrogate-ml.git)
cd ccs-squeeze-surrogate-ml
pip install -r requirements.txt
```

### 2. Run the full modeling and visualization pipeline

```bash
python run_experiment.py
```

The script automatically detects whether a CUDA-capable GPU is available for XGBoost and falls back to multi-threaded CPU execution otherwise. All CSV tables and 300 DPI figures are written to `./results/`.

## Reproduced Benchmark Results (Test Set, *n* = 500)

| Target Variable | Model | 5-Fold CV $R^2$ | Test $R^2$ | Test RMSE | Test MAE | Test MRPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Permeability Impairment, $\Delta k_{\text{imp}}$ (%) | Linear Regression | $0.7554 \pm 0.0087$ | $0.7416$ | $4.140\%$ | $2.893\%$ | $31.65\%$ |
| Permeability Impairment, $\Delta k_{\text{imp}}$ (%) | Random Forest | $0.9473 \pm 0.0069$ | $0.9514$ | $1.795\%$ | $1.097\%$ | $8.98\%$ |
| Permeability Impairment, $\Delta k_{\text{imp}}$ (%) | **XGBoost (Hist)** | **$0.9581 \pm 0.0025$** | **$0.9635$** | **$1.557\%$** | **$0.877\%$** | **$7.55\%$** |
| Protection Volume, $V_{\text{prot}}$ (kt $\text{CO}_2$, Log) | Linear Regression | $0.4775 \pm 0.1042$ | $0.4932$ | $947.38\text{ kt}$ | $373.46\text{ kt}$ | $67.88\%$ |
| Protection Volume, $V_{\text{prot}}$ (kt $\text{CO}_2$, Log) | Random Forest | $0.8790 \pm 0.0377$ | $0.9062$ | $407.59\text{ kt}$ | $137.03\text{ kt}$ | $25.36\%$ |
| Protection Volume, $V_{\text{prot}}$ (kt $\text{CO}_2$, Log) | **XGBoost (Hist)** | **$0.8749 \pm 0.0344$** | **$0.9620$** | **$259.45\text{ kt}$** | **$89.09\text{ kt}$** | **$21.73\%$** |

## Citation

If you use this code or dataset in your research, please cite the preprint and Zenodo archive:

```bibtex
@article{kaslivtsev2026ccs_squeeze,
  author  = {Kaslivtsev, Roman},
  title   = {Near-Wellbore Salt Precipitation and Geochemical Scaling During CO2 Injection into Depleted Reservoirs: Mechanisms, Surrogate Predictive Modeling, and Preventive Squeeze Strategies},
  year    = {2026},
  note    = {EarthArXiv Preprint, DOI: }
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.