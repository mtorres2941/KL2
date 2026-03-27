# KL2: Kernel Density Estimation and the Dirichlet Distribution for Uncertainty Quantification of Building Material Emissions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and data for the paper:

> Torres, M., & Srubar, W. V. (under review). Using kernel density estimation and the Dirichlet distribution for uncertainty quantification of building material emissions.

### Abstract

Whole-building life cycle assessment (wbLCA) is typically a deterministic process that uses single-value inputs, often from environmental product declarations (EPDs), to estimate a building's embodied carbon emissions. Several major sources of uncertainty are frequently overlooked, and existing methods of uncertainty quantification (UQ) are limited by overreliance on expert judgment and their use of unsubstantiated distributions that misrepresent irregular datasets. To address these shortcomings, this study introduces KL2, a UQ method in wbLCA that employs kernel density estimation and the Dirichlet distribution to quantify the expected range of embodied carbon for building materials. In this study, each KL2 input parameter is examined through increasingly complex hypothetical scenarios with different uncertainty characteristics. Then, as a proof-of-concept, KL2 is applied to a set of global structural steel EPDs. KL2 aims to enhance transparency in environmental impact data by identifying critical gaps that impede the development of more accurate, informative LCAs.

## Repository Structure

```
KL2/
├── KL2.ipynb                     # Main analysis notebook
├── src/
│   ├── funcs_kde2.py             # Core KL2 functions (kl2, kl2_plot, etc.)
│   └── funcs_unit_conversion.py  # Unit conversion utilities
├── data/
│   ├── EPDs_steelstructural_2025-01-09.xlsx   # Structural steel EPDs (n=63)
│   ├── EPDs_Steel_2025-02-20_n636.csv         # Broad steel EPD dataset (n=636)
│   ├── EPDs_StructuralSteel_2025-02-20_n63.csv # Structural steel EPD subset
│   ├── Steel_IndustryAverageEPDS.xlsx         # Industry-average steel EPDs
│   ├── WorldSteelReport_exp-2025-03-14_17_27_36.xlsx # World steel production data
│   ├── Bandwidth_vs_StDev.xlsx               # Bandwidth sensitivity analysis
│   └── CountryCodes.xlsx                     # Country code reference table
├── outputs/
│   └── figures/                  # All generated figures (FIG1–7, case studies)
└── archive/
    ├── VOID_*.ipynb              # Exploratory notebooks (not part of final analysis)
    └── ARCHIVE_CodeFromRick/     # Original MCMC/DPGMM exploration by collaborator
```

## Key Methods

**KL2** is a probabilistic UQ method built on two components:

- **Kernel Density Estimation (KDE)** with variable bandwidths — each EPD observation is represented as a kernel whose width reflects the product-level uncertainty of that specific EPD
- **Dirichlet distribution** — used to assign weights to each kernel based on quantities (e.g., tons of steel produced per country), enabling representativeness-weighted aggregation

The method is examined across four scenarios of increasing complexity:

| Scenario | Description |
|----------|-------------|
| 1 | All quantities and uncertainty values are known |
| 2 | Product-level uncertainty varies across EPDs |
| 3 | Quantities are known, distributions are inferred |
| 4 | An environmental value (EV) target is introduced |

A proof-of-concept is then demonstrated using a global dataset of structural steel EPDs weighted by national steel production volumes.

## Getting Started

### Requirements

Install dependencies using the provided environment file:

```bash
conda env create -f environment.yml
conda activate kl2
```

Or install with pip:

```bash
pip install numpy pandas scipy matplotlib seaborn arviz pymc mpltern tqdm xarray openpyxl
```

### Running the Analysis

Open and run `KL2.ipynb` from the repository root:

```bash
jupyter lab KL2.ipynb
```

The notebook is self-contained. All data files are in `data/` and all source functions are in `src/`.

## Core Functions (`src/funcs_kde2.py`)

| Function | Description |
|----------|-------------|
| `kl2(X, W, BW)` | Main KL2 estimator — returns the weighted KDE distribution |
| `kl2_plot(...)` | Visualization of the KL2 distribution |
| `weighted_quantile(X, W, q)` | Weighted quantile estimation |
| `weighted_std(X, W)` | Weighted standard deviation |
| `weighted_bw(X, W, method)` | Weighted bandwidth selection (Scott/Silverman) |
| `bw_dirichlet(X, W, alpha)` | Dirichlet-based bandwidth estimation |

## Data Sources

- **EPDs**: Retrieved from the [EC3 (Embodied Carbon in Construction Calculator)](https://buildingtransparency.org/ec3) database
- **Steel production**: [World Steel Association](https://worldsteel.org) annual production report
- **Industry averages**: Published industry-average EPDs for structural steel

## Citation

If you use this code or data, please cite:

> Torres, M., & Srubar, W. V. (under review). Using kernel density estimation and the Dirichlet distribution for uncertainty quantification of building material emissions.

A Zenodo DOI will be added upon publication.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

This work was conducted at the [Living Materials Laboratory](https://www.srubarlab.com), University of Colorado Boulder, under the supervision of Dr. Wil Srubar. The exploratory Bayesian/DPGMM work in `archive/ARCHIVE_CodeFromRick/` was contributed by a collaborator.
