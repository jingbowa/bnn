# Empirical Application: NYC Ride-Sharing Elasticities

This folder reproduces the empirical results in the paper (Section 5 and Online Appendix E): Figure 2, Table 6, Table A5, and the associated in-text statistics. The full pipeline is included, from the public raw trip records to the paper exhibits.

## Data

**Analysis data — construct separately.** Full market-level parquet files are not in this GitHub export. The CSV files in `outputs/step2_aggregated_markets/markets/samples/` show the schema only. Download the public TLC raw inputs and run steps 1–2 before estimation (step 3). The paper uses 7,462,961 market cells. Saved results and figures are included for inspection without re-estimation.

**Other included inputs:**
- `data/taxi_zone_lookup.csv`, `data/taxi_zones/` — TLC zone lookup and shapefile.
- `data/New_York_City_Regular_All_Formulations_Retail_Gasoline_Prices.csv` — NYC retail gasoline prices (EIA), used for auxiliary driver-cost variables.

**Raw data — not included (~5 GB); required to construct the analysis data with the step 1–2 code.** NYC Taxi & Limousine Commission High-Volume For-Hire Vehicle trip records, January and February 2024, from <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>. Download `fhvhv_tripdata_2024-01.parquet` and `fhvhv_tripdata_2024-02.parquet` and place them in `data/main_data/`.

## Setup

```
pip install -r requirements.txt
```

Tested with Python 3.10 on Windows 11 with an NVIDIA RTX 4090 (CUDA 11.8). The estimation scripts fall back to CPU automatically if CUDA is unavailable.

## Run

From this folder (`empirical/`). Run steps 1–2 to construct the full analysis files, then step 3. Small CSV samples are not the full estimation dataset.

```
# Step 1: process raw inputs (requires data/main_data/, see Data above)
python code_latest/step1_data_preparation/1_prepare_gasoline.py
python code_latest/step1_data_preparation/2_prepare_zones.py
python code_latest/step1_data_preparation/3_prepare_zone_distances.py
python code_latest/step1_data_preparation/4_prepare_trips.py

# Step 2: sample restrictions and market-level aggregation (Section 5.1)
python code_latest/step2_analysis_preparation/1_identify_high_traffic_zones.py
python code_latest/step2_analysis_preparation/2_aggregate_to_markets.py

# Step 3: elasticity estimation and figures
python code_latest/step3_elasticity_estimation/2b_estimate_zone_elasticity_manhattan.py
python code_latest/step3_elasticity_estimation/3_estimate_temporal_elasticity.py
python code_latest/step3_elasticity_estimation/2a_estimate_zone_elasticity_all_nyc.py
python code_latest/utilities/visualize_manhattan_elasticity_map.py
python code_latest/utilities/visualize_all_nyc_elasticity_map.py
```

The scripts locate the package root from their own path, so they can also be invoked from other working directories. Note: `4_prepare_trips.py` writes ~1.3 GB per month to `outputs/step1_processed_data/trips/` (neither full intermediate nor full step-2 parquet outputs are shipped).

## Code-to-exhibit mapping

| Paper exhibit | Script(s) | Output file(s) |
|---|---|---|
| Figure 2 (Manhattan elasticity map) | `2b_estimate_zone_elasticity_manhattan.py` then `visualize_manhattan_elasticity_map.py` | `outputs/figures/step3_fig_manhattan_elasticity_map.pdf`; per-zone estimates in `outputs/step3_elasticity_estimates/step3_elasticity_zones_manhattan.csv` |
| Table 6 (elasticity by time window) | `3_estimate_temporal_elasticity.py` | `outputs/step3_elasticity_estimates/step3_elasticity_temporal_periods.csv` |
| Table A5 (zone-level distribution, 97/73 zones) | `2a_estimate_zone_elasticity_all_nyc.py` then `visualize_all_nyc_elasticity_map.py` | `outputs/figures/step3_fig_all_nyc_pre_winsorization_all_zones.csv`, `..._quality_zones.csv` |
| Winsorization statistics quoted in Section 5 | `visualize_manhattan_elasticity_map.py` | `outputs/figures/step3_fig_manhattan_winsorization_stats.csv` |
| Sample statistics in Section 5.1 | `1_identify_high_traffic_zones.py` | `outputs/step2_aggregated_markets/zone_selection/step2_restriction_summary.csv` |

Saved estimates and figures in `outputs/` are upstream paper results. The full construction and estimation have not been rerun in the BNN package audit; cross-environment equality is not guaranteed.

## Upstream reproducibility notes

- Point estimates are deterministic: re-running `2b` reproduces the shipped per-zone elasticities exactly (verified bit-for-bit).
- Bootstrap confidence intervals (Table 6) are seeded (`RANDOM_SEED = 42`, incremented per time window) and reproduce exactly on the tested environment; across different GPUs/PyTorch builds, small floating-point differences may appear in the last digits.
- The subsampling size is `ZONE_TUNING_PARAMETER = 200` in `code_latest/step3_elasticity_estimation/config_quality_filters.py`; the estimator combines scales (m, 2m) for bias reduction.
- Rebuilding the analysis data from the raw records (steps 1-2) reproduces every estimation-relevant column exactly (prices, ride counts, times, wait times, instrument). The auxiliary `geo_miles` column (zone-centroid geodesic distance, not used in estimation) may differ at the ~1e-5 mile level depending on the installed `geographiclib` version.

Approximate runtimes on the tested machine: step 1 about 20 minutes (dominated by trip processing); step 2 about 20 minutes; 2b about 5 minutes; 3 about 15 minutes (100 bootstrap draws x 7 windows); 2a about 6 minutes; visualization scripts under 2 minutes. Each elasticity point estimate takes roughly 0.4-0.7 seconds on the GPU.
