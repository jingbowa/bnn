# Paper replication and source provenance

## JASA

The package implements the base estimator and two-scale regression estimator,
fixed-scale bootstrap, and delete-one jackknife. Its tests use independent finite
sample oracles. Original experiments and the R/C++ implementation remain available
in Patrick Vossler's [paper replication](https://github.com/patrickvossler18/tdnn_paper)
and [TDNN package](https://github.com/patrickvossler18/tdnn_package).
This release does not claim to rerun the complete original JASA simulation study.

## Elasticity

[`replication/elasticity`](../replication/elasticity) exports the final submission's
simulation and NYC empirical code, saved results, tables, and selected data inputs.
[`SOURCE_MANIFEST.json`](../replication/elasticity/SOURCE_MANIFEST.json) records each
source blob and the source repository commit. The exported source is from
Wang and Huang's *Scalable Just-in-Time Price Elasticity Estimation*.

The final application is NYC ride-sharing, January–February 2024 Uber trips.
The older cigarette application is not the current replication target.
Full raw and market-level parquet data are not in this GitHub export. Download
public TLC records and run the construction pipeline before empirical estimation.
Small CSV samples show the schema; they are not substitutes for the full sample.
Saved estimates allow the reported tables and charts to be inspected without a
new estimation run. The reconstruction and full estimation have not been rerun
as part of the focused package verification.

The upstream pickle distance cache is omitted; JSON distances are retained.
The construction script can regenerate the pickle locally if needed. README
data-availability claims were corrected to match the actual repository contents.
No manuscript review correspondence or private positioning notes are exported.

Simulation and empirical estimator copies intentionally retain different
application-specific finite-difference/first-stage settings. Do not replace them
with the new scalar wrapper when reproducing the paper without checking those
settings and documenting the change.
