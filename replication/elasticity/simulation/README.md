# Replication Package

**Paper:** *Scalable Just-in-Time Price Elasticity Estimation*

This package contains the simulation code and pre-computed results for all Monte Carlo tables and figures in the paper.

---

## Code-to-Manuscript Mapping

> **Note:** The internal script numbering reflects the original development order and does not match the final manuscript numbering. Use the tables below to find the script for any specific table or figure.

### Main Tables

| Manuscript | Description | Simulation Script | Formatting | Result File |
|---|---|---|---|---|
| **Table 1, Panel A** | Baseline RC logit models | `table_1_models.py` | `all_tables.py → table_1()` | `table_1_models.json` |
| **Table 1, Panel B** | Supply-side equilibrium | `table_5_supply.py` | `all_tables.py → table_5()` | `table_5_supply.json` |
| **Table 1, Panel C** | Complementarity & variety | `table_4_extra_models.py` | `all_tables.py → table_4()` | `table_4_extra_models.json` |
| **Table 2** | Implied optimal prices (baseline DGPs) | `table_6_baseline_pricing.py` | `all_tables.py → table_6_baseline()` | `table_6_baseline_pricing.json` |
| **Table 3** | Literature-calibrated DGPs | `table_6_literature.py` | `all_tables.py → table_6()` | `table_6_literature.json` |
| **Table 4, Panel A** | Computation time (J=4) | `table_2_scalability.py` | `all_tables.py → table_2()` | `table_2_scalability.json` |
| **Table 4, Panel B** | Computation time (J=20) | `table_2_scalability_20.py` | `all_tables.py → table_2_20()` | `table_2_scalability_20.json` |
| **Table 5** | MSE-minimizing tuning (own elasticity) | `table_3_tuning.py` + `table_3_tuning_extended.py` | `all_tables.py → table_3()` | `table_3_tuning_full.json`, `table_3_tuning_full_extended.json` |
| **Table 6** | Empirical: NYC ride-sharing elasticities | see `../empirical/` (`3_estimate_temporal_elasticity.py`) | — | `step3_elasticity_temporal_periods.csv` |

### Appendix Tables

| Manuscript | Description | Simulation Script | Formatting | Result File |
|---|---|---|---|---|
| **Table A1** | Parameter values for literature-calibrated DGPs | `calibrate_literature.py` | — (values entered in LaTeX) | `calibrated_params.json` |
| **Table A2** | Robustness (sample size & products) | `table_1_models.py` (alt. parameters) | — (values entered in LaTeX) | `table_1_models.json` |
| **Table A3, Panel A** | Tuning: cross-price elasticity | `table_3_tuning_extended.py` | `all_tables.py → table_3()` | `table_3_tuning_full_extended.json` |
| **Table A3, Panel B** | Tuning: reduced-form polynomial | `table_3_tuning_reduced_form.py` | `all_tables.py → table_3_reduced_form()` | `table_3_tuning_reduced_form.json` |
| **Table A4** | Extra primitive parameter values | `table_1_models.py` (alt. parameters) | — (values entered in LaTeX) | `table_1_models.json` |
| **Table A5** | Empirical: NYC zone-level elasticity distribution | see `../empirical/` (`2a_estimate_zone_elasticity_all_nyc.py`) | — | `step3_fig_all_nyc_pre_winsorization_*.csv` |
| *(unnumbered)* | PyBLP computation time | `table_appendix_pyblp.py` | `all_tables.py → table_app_pyblp()` | `table_appendix_pyblp.json` |
| *(unnumbered)* | BNN vs polynomial timing | `table_polynomial_time.py` | — (writes directly) | — |

### Figures

| Manuscript | Description | Script | Result File |
|---|---|---|---|
| **Figure 1(a)** | Elasticity profiles: correlated BLP | `figure_BLP.py` | `fig_elas_curve_BLP.json` |
| **Figure 1(b)** | Elasticity profiles: complementarity | `figure_complements.py` | `fig_elas_curve_complements.json` |
| **Figure 1(c)** | Elasticity profiles: variety & quantity | `figure_variety.py` | `fig_elas_curve_variety.json` |
| **Figure 2** | Empirical: Manhattan elasticity map | see `../empirical/` (`2b_...py` + `visualize_manhattan_elasticity_map.py`) | `step3_fig_manhattan_elasticity_map.pdf` |
| **Figure A2** | Optimal tuning visualization | `all_figures.py` | reads `tables/table_3_opt_tuning_own.csv` |

All scripts above are relative to `simulation_codes/`, `figures/`, or `tables/` as appropriate.

---

## Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- NVIDIA GPU with CUDA 11.8 support (optional; CPU-only execution is supported)

### Environment

Create and activate the conda environment. Two specifications are provided: `environment_ubuntu.yml` pins the exact versions of the Ubuntu server that generated the results; `environment_windows.yml` is a portable specification for other platforms.

```bash
conda env create -f environment_ubuntu.yml   # or environment_windows.yml
conda activate Price_elasticity
```

Key dependencies: Python 3.10, PyTorch 2.4, NumPy 1.26, Numba 0.60, SciPy 1.13, Pandas 2.2. The complementarity and variety/quantity DGPs (Table 1 Panel C and Figures 1b/1c) additionally require `NumbaMinpack` (`pip install NumbaMinpack`), which builds from source; we recommend installing it on Linux (it requires a C/Fortran toolchain on Windows).

---

## Directory Structure

```
replication/
├── README.md                   # This file
├── environment_ubuntu.yml      # Conda environment (exact versions, results server)
├── environment_windows.yml     # Conda environment (portable specification)
│
├── bnn_modules/                # Core estimator implementations
│   ├── bnn_torch.py            #   PyTorch implementation (GPU/CPU)
│   ├── bnn_numba.py            #   Numba-accelerated implementation (CPU)
│   └── bnn_module_supply.py    #   Supply-side variant
│
├── DGP_models/                 # Data-generating processes
│   ├── model_module_torch.py   #   Logit, independent RC logit, correlated RC logit
│   ├── model_module_extra.py   #   Complementarity and variety/quantity models
│   ├── model_module_blp.py     #   Parameterized BLP discrete choice DGP
│   └── model_module_rf.py      #   Reduced-form polynomial models
│
├── simulation_codes/           # Scripts that run Monte Carlo simulations
│   ├── table_1_models.py       #   → Table 1 Panel A
│   ├── table_2_scalability.py  #   → Table 4 Panel A
│   ├── table_2_scalability_20.py  # → Table 4 Panel B
│   ├── table_3_tuning.py       #   → Table 5
│   ├── table_3_tuning_extended.py #  → Table A3 Panel A
│   ├── table_3_tuning_reduced_form.py # → Table A3 Panel B
│   ├── table_4_extra_models.py #   → Table 1 Panel C
│   ├── table_5_supply.py       #   → Table 1 Panel B
│   ├── table_6_baseline_pricing.py # → Table 2
│   ├── table_6_literature.py   #   → Table 3
│   ├── calibrate_literature.py #   → Table A1 (one-time calibration)
│   ├── table_appendix_pyblp.py
│   └── table_polynomial_time.py
│
├── figures/                    # Scripts that generate figures
│   ├── figure_BLP.py           #   → Figure 1(a)
│   ├── figure_complements.py   #   → Figure 1(b)
│   ├── figure_variety.py       #   → Figure 1(c)
│   └── all_figures.py          #   → Figure A2 (tuning visualization)
│
├── tables/                     # Scripts that convert results to LaTeX
│   └── all_tables.py
│
└── simulation_results/         # Pre-computed simulation outputs (JSON)
    ├── table_1_models.json
    ├── table_2_scalability.json
    ├── table_2_scalability_20.json
    ├── table_3_tuning_full.json
    ├── table_3_tuning_full_extended.json
    ├── table_3_tuning_reduced_form.json
    ├── table_4_extra_models.json
    ├── table_5_supply.json
    ├── table_6_baseline_pricing.json
    ├── table_6_literature.json
    ├── calibrated_params.json
    ├── table_appendix_pyblp.json
    ├── fig_elas_curve_BLP.json
    ├── fig_elas_curve_complements.json
    └── fig_elas_curve_variety.json
```

---

## How to Reproduce

### Quick verification

The `simulation_results/` directory contains pre-computed JSON outputs from the Monte Carlo simulations. To verify that these results reproduce all manuscript tables, run a single command:

```bash
cd replication
python tables/all_tables.py
```

This reads the JSON files and produces `.tex` and `.csv` files in `tables/`. The generated values can be compared directly against the tables in the paper.

### Full replication (re-running simulations from scratch)

To regenerate the JSON result files themselves, run the simulation scripts in `simulation_codes/`. Each script is self-contained:

```bash
python simulation_codes/table_1_models.py
python simulation_codes/table_2_scalability.py
python simulation_codes/table_2_scalability_20.py
python simulation_codes/table_3_tuning.py
python simulation_codes/table_3_tuning_extended.py
python simulation_codes/table_3_tuning_reduced_form.py
python simulation_codes/table_4_extra_models.py
python simulation_codes/table_5_supply.py
python simulation_codes/table_6_baseline_pricing.py
python simulation_codes/table_6_literature.py
python simulation_codes/table_appendix_pyblp.py
python simulation_codes/table_polynomial_time.py
```

Results are saved as JSON files in `simulation_results/`, overwriting the pre-computed outputs. Then run `python tables/all_tables.py` to regenerate the LaTeX tables.

Note: `calibrate_literature.py`, `table_6_literature.py`, and `table_6_baseline_pricing.py` skip configurations already present in their output JSON files (a resume feature for long runs). To force full regeneration, delete or rename the corresponding file in `simulation_results/` first.

### Generating figures

```bash
python figures/figure_BLP.py
python figures/figure_complements.py
python figures/figure_variety.py
python figures/all_figures.py
```

Figures are saved as `.png` files in `figures/`.

---

## Runtime Notes

**Please read before re-running.** Verifying the shipped results (`python tables/all_tables.py` on the pre-computed JSONs) takes seconds. Re-running all simulations from scratch requires several days of total compute; the heaviest scripts are flagged below. Times refer to the results server (dual Xeon, RTX 3090) unless stated otherwise.

- **Table 1, Panel A** (`table_1_models.py`): 50 Monte Carlo iterations per model; minutes on GPU.
- **Table 1, Panel B** (`table_5_supply.py`): Solves equilibrium prices for all 20,000 markets per iteration via `joblib` (shipped `n_jobs=80`); adjust `n_jobs` to your core count. Expect roughly 4 minutes per Monte Carlo iteration on a 32-thread workstation, i.e., about 10 hours for the full 3 models × 50 iterations; scales with cores.
- **Table 1, Panel C** (`table_4_extra_models.py`): The variety/quantity model is the slowest single script (~12 hours on the server); requires `NumbaMinpack`.
- **Table 2** (`table_6_baseline_pricing.py`) and **Table 3** (`table_6_literature.py`): Equilibrium pricing across many configurations; can take several hours each. Both resume from configurations already present in their output JSONs (see the note under "Full replication"). Table 3 uses pre-calibrated parameters stored in `calibrated_params.json`; to re-run calibration from scratch: `python simulation_codes/calibrate_literature.py`.
- **Table 4** (`table_2_scalability.py` / `table_2_scalability_20.py`): Benchmarks Numba CPU, PyTorch CPU, and PyTorch GPU; requires a GPU for full replication. Reported times are hardware-specific — expect qualitative, not numerical, agreement on different machines.
- **Table 5 / Table A3** (`table_3_tuning.py`, `table_3_tuning_extended.py`, `table_3_tuning_reduced_form.py`): 1,000 Monte Carlo iterations per tuning value and product count — the most compute-intensive tables (many hours to days in total). GPU strongly recommended.
- **Figures 1(a)–1(c)** (`figures/figure_*.py`): Each traces elasticity curves over a price grid with many Monte Carlo draws; allow an hour or more per figure. Figures 1(b)/1(c) require `NumbaMinpack`.
- **PyBLP appendix** (`table_appendix_pyblp.py`): Requires the `pyblp` package (install via `pip install pyblp`).

---

## Hardware

Results in the paper were generated on a workstation running Ubuntu 22.04.3 LTS, equipped with dual Intel Xeon Gold 6348 CPUs and a single NVIDIA GeForce RTX 3090 GPU.
