# Replication Package

**Paper:** *Scalable Just-in-Time Price Elasticity Estimation*
**Authors:** Jingbo Wang (Chinese University of Hong Kong) and Yufeng Huang (University of Rochester)
**Contact:** jingbowang@cuhk.edu.hk

This package has two self-contained components:

- **`simulation/`** — all Monte Carlo results: Tables 1-5, Tables A1-A4, Figures 1 and A2. See `simulation/README.md` for the full code-to-exhibit mapping, setup, and pre-computed results.
- **`empirical/`** — the NYC ride-sharing application: Figure 2, Table 6, Table A5, and associated in-text statistics. See `empirical/README.md`.

Each component includes its own copy of the bagged nearest neighbors (BNN) estimator modules (`bnn_modules/`); the two copies differ intentionally in application-specific settings (finite-difference step size and the first-stage derivative approximation) and should not be merged.

## Data availability

- The simulation component generates all data (no external data used).
- The empirical component uses public NYC Taxi & Limousine Commission trip records (High-Volume For-Hire Vehicle, license HV0003, January-February 2024), freely available at <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>. The full construction pipeline (raw records to analysis data, following Section 5.1 of the paper) is included, and the construction scripts and small CSV schema samples are included. Full market-level parquet files are not shipped in this GitHub export; construct them from the raw records before estimation.

## Environments

Both components run on Python 3.10. Pinned dependencies are listed in each component (`simulation/environment_*.yml`, `empirical/requirements.txt`). A CUDA-capable GPU is recommended for the empirical component (tested on an NVIDIA RTX 4090) but not required.
