# BNN — bagged nearest neighbors

Nonparametric estimation through the distribution of nearest neighbors: a Python
package, algorithm notes, reproducible examples, and paper replication materials.
Research resource: [jingbowa.github.io/bnn](https://bnn-production-9654.up.railway.app/).

Explore: [theory](docs/theory.md) · [algorithms](docs/algorithms.md) ·
[replication](docs/replication.md) · [verification](docs/verification.md).

BNN averages nearest-neighbor predictions over subsamples drawn **without
replacement**, evaluated through exact rank weights. It is the **distributional
nearest-neighbor (DNN)** estimator in the JASA paper. **Two-scale DNN (TDNN)**
is its bias-corrected extension. BNN here is not a Bayesian neural network;
DNN here is not a deep neural network.

## Papers and development

- Demirkaya, Fan, Gao, Lv, Vossler, and Wang (2024).
  [Optimal Nonparametric Inference with Two-Scale Distributional Nearest Neighbors](https://doi.org/10.1080/01621459.2022.2115375).
  *Journal of the American Statistical Association*, 119(545), 297–307.
- Wang and Huang (2026). [Scalable Just-in-Time Price Elasticity Estimation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3557359).
  *Management Science*, forthcoming. Final application: NYC ride-sharing.
- Earlier working paper: Fan, Lv, and Wang (2018),
  [DNN: A Two-Scale Distributional Tale of Heterogeneous Treatment Effect Inference](https://ssrn.com/abstract=3238897).

The bagging construction builds on earlier nearest-neighbor work, including
Steele (2009) and Biau, Cérou, and Guyader (2010). See [references](docs/references.bib).
The current implementation is developed from the elasticity code. Patrick
Vossler's [TDNN R/C++ package](https://github.com/patrickvossler18/tdnn_package)
and [JASA replication](https://github.com/patrickvossler18/tdnn_paper) are companion
resources inspected for comparison.

## Install

Python 3.10+ and NumPy 1.24+:

```sh
git clone https://github.com/jingbowa/bnn.git
cd bnn
python -m pip install .
python examples/regression.py
```

Install from source; a PyPI distribution has not been published.

## Regression and inference

```python
import numpy as np
from bnn import BNN, TwoScaleBNN

X = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 1.]])
y = np.array([0., 1., 1., 2., 3.])
base = BNN(s=2).fit(X, y)
model = TwoScaleBNN(s1=2, s2=4).fit(X, y)
prediction = model.predict([[.5, .5]])
bootstrap = model.bootstrap([[.5, .5]], n_resamples=499, random_state=42)
print(prediction, bootstrap.standard_error)
print(model.jackknife([[.5, .5]]).standard_error)
```

Scales are integers: `1 <= s <= n`, or `1 <= s1 < s2 <= n`. Fit copies training
arrays and caches weights. Predictions have shape `(n_queries,)`. Features and
responses must be real and finite; `y` is one-dimensional. Apply standardization
consistently before fitting and predicting. Exact distance ties preserve original
training-row order. Negative two-scale weights are intentional; predictions can
leave the observed response range.

Bootstrap uses paired draws across queries and sample variance with denominator
`B-1`; `samples` has shape `(B, n_queries)`. Jackknife follows the JASA formula
centered on the original estimate and requires every scale smaller than `n`.
Both keep scales fixed and omit scale-selection uncertainty. Valid inference
depends on the paper's assumptions.

`select_scales(X_train, y_train, X_validation, y_validation, candidates)` minimizes
global held-out MSE over supplied `(s1,s2)` pairs, sharing each query's ordering.
This helper differs from the paper's pointwise tuning algorithm.

## Accelerator prediction

```sh
python -m pip install '.[torch]'   # or '.[jax]'
```

```python
from bnn.torch_backend import TorchBNN, TorchTwoScaleBNN
model = TorchTwoScaleBNN(2, 4, device="cuda", batch_size=8).fit(X, y)
prediction = model.predict([[.5, .5]])  # tensor on the selected device
```

```python
import jax
jax.config.update("jax_enable_x64", True)
from bnn.jax_backend import JAXBNN, JAXTwoScaleBNN
model = JAXTwoScaleBNN(2, 4, device=jax.devices()[0], batch_size=8).fit(X, y)
prediction = model.predict([[.5, .5]])  # JAX array
```

| Backend | Checked execution | Scope |
|---|---|---|
| NumPy | CPU, float64 | Prediction, regression bootstrap/jackknife, validation selection |
| PyTorch | CPU and CUDA, float32/float64 | Batched prediction |
| JAX | CPU, float32/float64 | JIT/batched prediction; GPU and TPU hardware remain unverified |

Use NumPy for regression resampling. Extras alone do not configure CUDA/TPU
drivers: follow official [PyTorch](https://pytorch.org/get-started/locally/) and
[JAX](https://docs.jax.dev/en/latest/installation.html) instructions. Float64 is
the default; JAX requires explicit precision opt-in. Float32 can change ordering
at nearly equal distances. Batching bounds workspace to `O(batch_size*n*d)`.
The first JAX call includes compilation; different shapes may recompile.
Legacy one-query adapters remain available:
`bnn.torch_backend.bnn_2scale_torch(y,X,x,s1)` and
`bnn.jax_backend.bnn_2_jax(y,X,x,s1)`; omitted `s2` means `2*s1`.

## Scalar price elasticity

```python
from bnn import PriceElasticity, TwoScaleBNN
model = PriceElasticity(
    TwoScaleBNN(7, 14), price_index=0, instrument_index=-1, step=0.1
).fit(X, quantity)
result = model.predict(X_query)
print(result.elasticity, result.uncorrected_elasticity)
```

This extension implements **one endogenous price and one instrument**. The
default first stage regresses price on intercept, `z`, and `z**2` using stable
least squares. Optional `first_stage_controls` enter linearly. `step` is the
total span of a centered difference. First stage and rank weights fit once;
five prediction blocks share the backend. Torch/JAX regressors can be supplied
with NumPy training inputs. Results and first-stage least squares use the CPU.
Elasticity bootstrap remains in paper replication, rather than this wrapper.

The correction `(g_p+g_z/r_z)*p/g` requires the paper's instrument/structural
assumptions, nonzero first-stage derivative, and positive predicted quantity.
Numerical checks alone do not establish identification. See
[algorithms](docs/algorithms.md) and [replication](docs/replication.md).

## Computation and verification

Recurrence weights avoid large log-gamma subtractions and are cached at fit.
One stable sort serves both scales. No arbitrary tail cutoff or approximate
neighbor search is used. CPU queries cost `O(n*d+n*log(n))` time and `O(n*d)`
workspace. Bootstrap reuses ordering through resampling counts with an exact
stable-sort fallback for ties. Jackknife prefix/suffix sums avoid `n` refits.

```sh
python -m unittest discover -s tests -v
python -m pip install '.[benchmark]'
python benchmarks/benchmark.py
python examples/figures.py
```

Independent tests cover exact binomial weights, exhaustive subset averages,
explicit resampling/deletion, known structural derivatives, and backends.
[Verification](docs/verification.md) records measured workloads, versions,
baselines, and limits. The complete original R/C++ package and full NYC
estimation have not been rerun in this focused verification.

## Replication and citation

[`replication/elasticity`](replication/elasticity) contains final simulation and
NYC code with saved results, tables, and source provenance. Full trip and
market-level parquet data must be obtained/constructed separately. CSV samples
show the schema. Saved figures are paper results, not new estimates.

Cite JASA for two-scale estimation/inference; also cite Wang and Huang for the
endogeneity extension or NYC application. [BibTeX](docs/references.bib) and
[CITATION.cff](CITATION.cff) include software and paper references.

Implementation and maintenance: [Jingbo Wang](https://jingbowa.github.io).
Scientific contributions are credited through each paper's full author list.
