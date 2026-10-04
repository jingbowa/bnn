# Bagged nearest neighbors

BNN averages nearest-neighbor responses over all size-`s` subsets drawn without
replacement. Sorting by distance gives the exact representation

`estimate(x) = sum_i w_i(n,s) y_(i)`,
`w_i(n,s) = choose(n-i,s-1)/choose(n,s)`.

The recurrence starts at `w_1=s/n` and uses
`w_(i+1)=w_i*(n-i-s+1)/(n-i)` over the nonzero support. No subsamples are
enumerated. The endpoint `s=1` gives the sample mean; `s=n` gives nearest neighbor.

For two-scale BNN, dimension `d` and actual integer scales `s1<s2` give
`a1=-1/((s2/s1)**(2/d)-1)` and `a2=1-a1`. Use combined rank weights
`a1*w(n,s1)+a2*w(n,s2)` with one distance ordering. Negative weights are intended.

| Routine | Steps | Computational reuse |
|---|---|---|
| Predict | Compute distances, stable-sort, weighted sum | Fit caches weights; each query ordering serves both scales |
| Bootstrap | Paired sample resampling, fixed scales, sample SD | Counts and cumulative weights reuse the original ordering; tied distances use stable-sort fallback |
| Jackknife | Delete each observation; variance centered on original estimate | Prefix/suffix sums avoid refitting and sorting n times |
| Held-out scale selection | Score supplied pairs by global validation MSE | One ordering per validation query |
| Scalar elasticity | Fit first stage, evaluate five query blocks, form derivative correction | First stage and BNN weights fit once |

For scalar elasticity write `g(p,z,c)=E[quantity|p,z,c]` and
`r(z,c)=E[price|z,c]`. The correction is `(g_p+g_z/r_z)*p/g`, under the
elasticity paper's identification conditions. The wrapper estimates `g_p` and
`g_z` using centered differences of total span `step`. The default first stage
is least squares on intercept, `z`, and `z**2`; optional controls enter linearly.
The wrapper checks first-stage rank, near-zero `r_z`, and positive prediction.

Do not interpret estimator predictions as structural effects without the paper's
identification assumptions. A nonsingular first stage does not prove instrument
validity. Finite differences of a rank estimator are numerical derivative
estimates; the estimator is not an automatic-differentiation neural network.

See [references](references.bib) for the papers and earlier foundations.
