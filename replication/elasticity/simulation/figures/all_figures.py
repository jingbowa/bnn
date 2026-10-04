import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_optimal_tuning_own():
    # Read the data
    df = pd.read_csv('./tables/table_3_opt_tuning_own.csv')

    # Convert wide to long format for better plotting
    id_vars = ['model', 'observations']
    value_vars = [str(x) for x in range(4, 22, 2)]
    df_long = pd.melt(df, 
                    id_vars=id_vars,
                    value_vars=value_vars,
                    var_name='tuning_parameter',
                    value_name='value')
    df_long['tuning_parameter'] = df_long['tuning_parameter'].astype(int)

    # Set style
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Create figure with custom size
    fig, ax = plt.subplots(figsize=(10, 7))  # Slightly smaller figure since legend is inside

    # Define distinctive colors for sample sizes using different hues
    colors = {
        10000: '#1f77b4',  # Blue
        20000: '#ff7f0e',  # Orange 
        40000: '#2ca02c',  # Green
        60000: '#d62728'   # Red
    }

    # Define markers and linestyles for models
    markers = {
        'BLP_corr': 'o',   # Circle
        'BLP_ind': 's',    # Square
        'Logit': '^'       # Triangle
    }

    linestyles = {
        'BLP_corr': '-',    # Solid
        'BLP_ind': '--',    # Dashed
        'Logit': '-.'       # Dash-dot
    }

    # Plot each combination of model and sample size
    for obs in sorted(df_long['observations'].unique()):
        for model in sorted(df_long['model'].unique()):
            data = df_long[(df_long['model'] == model) & (df_long['observations'] == obs)]
            plt.plot(data['tuning_parameter'], data['value'],
                    label=f'n={obs:,}, {model}',
                    color=colors[obs],
                    marker=markers[model],
                    linestyle=linestyles[model],
                    markersize=6,  # Slightly smaller markers
                    linewidth=1.5,  # Slightly thinner lines
                    alpha=0.8,
                    markerfacecolor='white',
                    markeredgewidth=1.5,
                    markeredgecolor=colors[obs])

    # Customize the plot
    plt.xlabel('Tuning Parameter', fontsize=12, fontweight='bold')
    plt.ylabel('Value', fontsize=12, fontweight='bold')
    plt.title('Optimal Tuning Parameters Across Models and Sample Sizes', 
            fontsize=14, fontweight='bold', pad=20)

    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.5)
    ax.set_axisbelow(True)

    # Create a more organized legend with sample size groups
    handles, labels = ax.get_legend_handles_labels()
    # Group by sample size
    sample_groups = {}
    for h, l in zip(handles, labels):
        sample_size = l.split(',')[0]
        if sample_size not in sample_groups:
            sample_groups[sample_size] = []
        sample_groups[sample_size].append((h, l))

    # Create legend with spacing between sample size groups
    legend_handles = []
    legend_labels = []
    for sample_size in sorted(sample_groups.keys()):
        for h, l in sample_groups[sample_size]:
            legend_handles.append(h)
            legend_labels.append(l)
        # Add a blank entry for spacing between groups
        legend_handles.append(plt.Line2D([0], [0], alpha=0))
        legend_labels.append('')

    # Add legend inside the plot with semi-transparent background
    legend = ax.legend(legend_handles, legend_labels,
                    loc='upper left',
                    bbox_to_anchor=(0.02, 0.98),  # Fine-tune position
                    borderaxespad=0.,
                    fontsize=12,  # Smaller font size
                    frameon=True,
                    edgecolor='black',
                    fancybox=True)  # Rounded corners
    legend.get_frame().set_alpha(0.9)  # Semi-transparent background
    legend.get_frame().set_facecolor('white')  # White background

    # Set background color
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.5)

    # Set x-axis ticks to match actual tuning parameter values
    plt.xticks(df_long['tuning_parameter'].unique())

    # Add subtle box around plot area
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Add secondary y-axis ticks
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim(ax.get_ylim())  # match the limits of the primary axis
    ax2.tick_params(axis='y', labelright=True, labelleft=False)  # show ticks on right, hide labels on left
    ax.tick_params(axis='y', labelright=False, labelleft=True)   # show ticks on left, hide labels on right

    # Adjust layout
    plt.tight_layout()

    # Save the figure with high DPI
    plt.savefig('./figures/optimal_tuning_own.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    return None

if __name__ == '__main__':
    plot_optimal_tuning_own()