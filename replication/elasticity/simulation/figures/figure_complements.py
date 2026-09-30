import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt

from bnn_modules.bnn_numba import p_elas, p_elas_boot
from DGP_models.model_module_extra import Complements_data, Complements_stats

def compute_elasticities(
    model_data=Complements_data,
    model_stats=Complements_stats,
    monte=10,
    bins=41,
    tuning_s=7,
    price_range=(0.3, 0.7)
):
    # Pre-compute price points
    p = np.linspace(price_range[0], price_range[1], bins)
    
    # Initialize arrays
    results = {
        'e_own': np.zeros((bins, monte)),
        'var_own': np.zeros((bins, monte)),
        'e_cross': np.zeros((bins, monte)),
        'var_cross': np.zeros((bins, monte)),
        't_own': np.zeros((bins, monte)),
        't_cross': np.zeros((bins, monte))
    }
    
    # Monte Carlo iterations
    for j in range(monte):
        y1, y2, XZ, p1, z = model_data()
        
        for i, p_eval in enumerate(p):
            print(f"Processing iteration {j+1}/{monte}, price point {i+1}/{bins}")
            xz = np.array([p_eval, 0.5, 0.5, 0])
            
            # Compute elasticities and variances
            results['e_own'][i, j] = p_elas(y1, XZ, xz, tuning_s, p1, z)
            _, results['var_own'][i, j] = p_elas_boot(y1, XZ, xz, tuning_s, p1, z)
            
            results['e_cross'][i, j] = p_elas(y2, XZ, xz, tuning_s, p1, z)
            _, results['var_cross'][i, j] = p_elas_boot(y2, XZ, xz, tuning_s, p1, z)
            
            results['t_own'][i, j], results['t_cross'][i, j] = model_stats(np.array([p_eval, 0.5, 0.5]))

    # Compute summary statistics
    processed_results = {
        'p': p.tolist(),
        'e_own': np.mean(results['e_own'], axis=1).tolist(),
        't_own': np.mean(results['t_own'], axis=1).tolist(),
        'std_own': np.sqrt(np.mean(results['var_own'], axis=1)).tolist(),
        'e_cross': np.mean(results['e_cross'], axis=1).tolist(),
        't_cross': np.mean(results['t_cross'], axis=1).tolist(),
        'std_cross': np.sqrt(np.mean(results['var_cross'], axis=1)).tolist()
    }

    # Save results
    with open('./simulation_results/fig_elas_curve_complements.json', 'w') as f:
        json.dump(processed_results, f)

    return processed_results

def plot_elasticity_curves():
    with open('./simulation_results/fig_elas_curve_complements.json', 'r') as f:
        results = json.load(f)

    # Convert lists to numpy arrays
    p = np.array(results['p'])
    e_own = np.array(results['e_own'])
    t_own = np.array(results['t_own'])
    std_own = np.array(results['std_own'])
    e_cross = np.array(results['e_cross'])
    t_cross = np.array(results['t_cross'])
    std_cross = np.array(results['std_cross'])
    
    # Set style for professional plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    
    # Color scheme
    estimate_color = '#2E86C1'
    true_color = '#E74C3C'
    ci_color = '#AED6F1'

    # Plot own elasticity
    ax[0].plot(p, e_own, marker="^", color=estimate_color, label='Estimated', markersize=8, linewidth=2)
    ax[0].plot(p, t_own, marker="s", color=true_color, linestyle="--", label='True', markersize=8, linewidth=2)
    ax[0].fill_between(p, e_own + 1.96*std_own, e_own - 1.96*std_own, color=ci_color, alpha=0.3, label='95% CI')

    # Plot cross elasticity
    ax[1].plot(p, e_cross, marker="^", color=estimate_color, label='Estimated', markersize=8, linewidth=2)
    ax[1].plot(p, t_cross, marker="s", color=true_color, linestyle="--", label='True', markersize=8, linewidth=2)
    ax[1].fill_between(p, e_cross + 1.96*std_cross, e_cross - 1.96*std_cross, color=ci_color, alpha=0.3, label='95% CI')

    # Common styling for both subplots
    for i in range(2):
        ax[i].set_xlabel('Price Level', fontsize=12, fontweight='bold')
        ax[i].grid(True, alpha=0.3)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].tick_params(axis='both', which='major', labelsize=10)
        ax[i].legend(fontsize=10, frameon=True, fancybox=True, shadow=True)

    # Specific styling for each subplot
    ax[0].set_ylabel('Own Price Elasticity', fontsize=12, fontweight='bold')
    ax[0].set_ylim((-1.9, 0))
    ax[0].set_xlim(0.3, 0.7)

    ax[1].set_ylabel('Cross Price Elasticity', fontsize=12, fontweight='bold')
    ax[1].set_ylim(-0.5, 0.5)
    ax[1].set_xlim(0.3, 0.7)

    # Add title and adjust layout
    fig.suptitle('Price Elasticity Analysis - Complements', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    # Save and close
    fig.savefig('./figures/fig_complements.png', bbox_inches="tight", dpi=300)
    plt.close()
    return fig

if __name__ == "__main__":
    compute_elasticities()
    plot_elasticity_curves()
