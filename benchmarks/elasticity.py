"""Check the scalar extension against the original elasticity implementation.

Set BNN_ELASTICITY_TORCH to the original bnn_torch.py file. CUDA timing is
synchronized. This workload measures point estimates, not bootstrap inference.
"""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from bnn import PriceElasticity, TwoScaleBNN
from bnn.torch_backend import TorchTwoScaleBNN

torch.set_num_threads(4)
device = 'cuda' if os.environ.get('BNN_TEST_CUDA') else 'cpu'
path = os.environ['BNN_ELASTICITY_TORCH']
spec = importlib.util.spec_from_file_location('reference', path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
rng = np.random.default_rng(137)
z = rng.uniform(-1, 1, 5000)
p = 2 + 1.5*z + .25*z*z + rng.normal(0, .2, len(z))
X = np.column_stack([p, rng.uniform(-1, 1, len(z)), z])
y = 20 - 2*p + 3*(p - (2 + 1.5*z + .25*z*z)) + rng.normal(0, .1, len(z))
Q = np.array([[2., .2, 0.], [2.8, -.2, .5], [1.5, .1, -.3]])
tx, ty, tq = [torch.tensor(a, dtype=torch.float64, device=device) for a in (X, y, Q)]
model = PriceElasticity(TorchTwoScaleBNN(7, 14, device=device)).fit(X, y)
def original():
    return torch.stack([reference.p_elas_2scale_torch(ty, tx, q, 0, -1, 7) for q in tq]).cpu().numpy().reshape(-1)
expected = original()
actual = model.predict(Q).elasticity
cpu = PriceElasticity(TwoScaleBNN(7, 14)).fit(X, y).predict(Q).elasticity
np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
np.testing.assert_allclose(actual, cpu, atol=1e-10, rtol=1e-10)
def timed(fn):
    fn()
    times = []
    for _ in range(5):
        if device == 'cuda': torch.cuda.synchronize()
        begin = time.perf_counter()
        fn()
        if device == 'cuda': torch.cuda.synchronize()
        times.append(time.perf_counter() - begin)
    return statistics.median(times)
result = {'device': device, 'n': len(X), 'd': 3, 'queries': 3, 's1': 7, 's2': 14,
          'step': .1, 'dtype': 'float64', 'threads': 4, 'repeats': 5,
          'max_abs_error_reference': float(np.max(np.abs(actual - expected))),
          'max_abs_error_numpy': float(np.max(np.abs(actual - cpu))),
          'baseline_seconds': timed(original),
          'optimized_seconds': timed(lambda: model.predict(Q)),
          'baseline': 'original p_elas_2scale_torch, refits quadratic first stage per query'}
Path('docs').mkdir(exist_ok=True)
Path(f'docs/elasticity_{device}.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
