"""Generate the research-page figures on a compute host.

The temporal chart redraws saved paper results; it does not re-estimate them.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bnn import BNN, TwoScaleBNN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'figure.facecolor': 'white'})
rng = np.random.default_rng(2026)
X = rng.uniform(-1, 1, (600, 1))
y = np.sin(3*X[:, 0]) + rng.normal(0, .2, len(X))
Q = np.linspace(-.95, .95, 201)[:, None]
fig, ax = plt.subplots(figsize=(8.5, 4.5), layout='constrained')
ax.scatter(X[:, 0], y, s=8, alpha=.2, color='#5a6a7a', label='Generated observations')
ax.plot(Q[:, 0], np.sin(3*Q[:, 0]), color='#2c3e50', linestyle='--', label='True mean')
ax.plot(Q[:, 0], BNN(20).fit(X, y).predict(Q), color='#a07a4c', label='BNN, s = 20')
ax.plot(Q[:, 0], TwoScaleBNN(20, 40).fit(X, y).predict(Q), color='#1a3a5c', label='Two-scale BNN, 20 / 40')
ax.set(xlabel='Feature x', ylabel='Response / conditional mean')
ax.legend(frameon=False, loc='upper left', fontsize=9)
fig.savefig(OUT / 'regression.svg')
plt.close(fig)
path = ROOT / 'replication/elasticity/empirical/outputs/step3_elasticity_estimates/step3_elasticity_temporal_periods.csv'
with path.open() as handle:
    rows = list(csv.DictReader(handle))
estimate, lo, hi = [np.array([float(r[k]) for r in rows]) for k in ('elasticity', 'ci_lower', 'ci_upper')]
fig, ax = plt.subplots(figsize=(8.5, 4.5), layout='constrained')
positions = np.arange(len(rows))[::-1]
ax.errorbar(estimate, positions, xerr=[estimate-lo, hi-estimate], fmt='o',
            capsize=4, color='#1a3a5c', ecolor='#5a6a7a')
ax.axvline(0, color='#a07a4c', linewidth=1, linestyle='--')
ax.set(yticks=positions, yticklabels=[r['period'] for r in rows],
       xlabel='Price elasticity (estimate and 95% confidence interval)')
fig.savefig(OUT / 'temporal.svg')
plt.close(fig)
print('Created regression.svg and temporal.svg')
