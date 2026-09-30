import numpy as np
from bnn import BNN, TwoScaleBNN

rng = np.random.default_rng(2026)
X = rng.uniform(0, 1, size=(500, 1))
y = np.sin(2 * np.pi * X[:, 0]) + rng.normal(0, 0.15, 500)
queries = np.linspace(0.1, 0.9, 9).reshape(-1, 1)

base = BNN(s=20).fit(X, y)
model = TwoScaleBNN(s1=20, s2=40).fit(X, y)
prediction = model.predict(queries)
bootstrap = model.bootstrap(
    queries, n_resamples=199, random_state=42
)
jackknife = model.jackknife(queries)

print(prediction)
print(bootstrap.standard_error)
print(jackknife.standard_error)
