"""Small reproducible example; execute on the compute host."""
import numpy as np
from bnn import TwoScaleBNN, select_scales

rng = np.random.default_rng(2026)
X = rng.uniform(-1, 1, (600, 2))
y = np.sin(X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, .15, len(X))
choice = select_scales(X[:450], y[:450], X[450:], y[450:], [(6, 12), (10, 20), (15, 30)])
model = TwoScaleBNN(choice.s1, choice.s2).fit(X, y)
query = np.array([[0., 0.], [.3, -.2]])
result = model.bootstrap(query, n_resamples=199, random_state=17)
print("Selected scales:", choice.s1, choice.s2)
print("Estimates:", result.estimate)
print("Bootstrap standard errors:", result.standard_error)
print("Jackknife standard errors:", model.jackknife(query).standard_error)

