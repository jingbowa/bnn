# Verification record

2026-09-30. Numerical work ran on the tower and gpu142. The Mac was the cockpit.

## Independent checks

30 unittest methods cover exact binomial rank weights, exhaustive subset
averages (base and two-scale), endpoints, arbitrary dimensions/scales, constant
responses, stable ties, large common coordinate offsets, explicit bootstrap,
delete-one jackknife, held-out selection, known structural correction, validation,
and Torch/JAX parity in float32/float64.

Separate CPU environments check NumPy/JAX and NumPy/PyTorch. The guarded CUDA
run checks NumPy and PyTorch including both precision paths. Optional backend
tests skip when their runtime is absent; the combined runs exercise all methods.
JAX GPU and TPU hardware are not tested.

The scalar elasticity comparison uses the original `p_elas_2scale_torch` routine
and the NumPy wrapper. CPU/CUDA discrepancies from the original are below 4e-13
on the recorded workload. Original one-element tensor returns are normalized
to one scalar per query. Stable least squares replaces normal equations;
behavior can differ on ill-conditioned first-stage designs.

## Prediction timings

64 queries; n=20,000; d=4; scales 20/40; float64; seed=71 for backend comparisons;
batch size 8; four CPU threads; one warmup; median of five runs. Fit, input-device
allocation and input transfer are excluded. CUDA boundaries synchronize; JAX
completion is awaited. Raw JSON records below contain exact measurements.

| Implementation | Seconds | Baseline / first-call time |
|---|---:|---|
| NumPy CPU | 0.086678 | Scalar NumPy/SciPy log-gamma formulation; not actual Numba runtime |
| PyTorch CPU | 0.029468 | Actual original scalar calls: 0.133419 |
| PyTorch CUDA RTX 3090 | 0.002393 | Actual original scalar calls: 0.039094 |
| JAX CPU, warmed | 0.037888 | First call with compilation: 0.131661 |

The NumPy workload uses seed 20260930 and a different generated sample. Its
baseline is explicitly an independent formulation. Measurements describe these
machines/workloads. Recorded CUDA allocated tensor memory peaks at 26.62 MiB.
Backend versions: CPU torch 2.14.0+cu130, CUDA torch 2.5.1+cu121, JAX 0.11.2.

Additional CPU records cover weight construction, held-out grid reuse,
bootstrap, and jackknife. Baselines are described per record; refit speedups
must not be generalized to already optimized external packages.

## Source and comparison

Correctness specification: JASA equations 6, 9–11, 24 and 26. Development base:
the elasticity NumPy/Numba and PyTorch routines. The final replication export
records source blobs/commit in replication/elasticity/SOURCE_MANIFEST.json.
Patrick Vossler's R/C++ source was inspected; complete R execution was not run.
For rounded second scales, coefficients here use the actual integer ratio,
preserving algebraic leading-bias cancellation.

## Limits

The synthetic illustration does not rerun the full JASA study. NYC charts use
saved final paper results; full construction and estimation have not been rerun.
Full parquet files are obtained separately. Historical paper timing code lacks
explicit GPU synchronization and is kept distinct from synchronized new checks.
Numerical checks establish finite-sample implementation behavior; they do not
establish instrument validity, theorem proofs, or arbitrary-data coverage.

## Raw records and scripts

- benchmark_cpu.json / benchmarks/benchmark.py
- benchmark_cpu_backends.json and benchmark_cuda_backends.json / benchmarks/backends.py
- benchmark_jax.json / benchmarks/jax_benchmark.py
- elasticity_cpu.json and elasticity_cuda.json / benchmarks/elasticity.py
- examples/figures.py generates regression and redraws saved temporal estimates.
