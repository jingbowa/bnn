import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json

def table_1():
    all_results = json.load(open("./simulation_results/table_1_models.json", "r"))
    
    rows = []
    for model, results in all_results.items():
        df = pd.DataFrame(results)
        for elasticity_type in ['own', 'cross']:
            true_value = df[f'true_{elasticity_type}'].mean()
            est_value = df[f'est_{elasticity_type}'].mean()
            rows.append({
                'model': model,
                'elasticity_type': elasticity_type,
                'true_value': true_value,
                'mean_estimate': est_value,
                'bias': est_value - true_value,
                'mean_se': df[f'se_{elasticity_type}'].mean(),
                'std_estimate': df[f'est_{elasticity_type}'].std()
            })
    
    summary_df = pd.DataFrame(rows)
    
    # Set 'model' and 'elasticity_type' as index
    summary_df.set_index(['model', 'elasticity_type'], inplace=True)
    
    # Save the summary table
    summary_df.to_latex("./tables/table_1_models.tex", float_format="%.3f")
    
    return summary_df

def table_2():
    results = json.load(open("./simulation_results/table_2_scalability.json", "r"))
    
    summary_df = pd.DataFrame(results)
    summary_df.set_index('Sample Size', inplace=True)
    
    # Save the summary table
    summary_df.to_latex("./tables/table_2_scalability.tex", float_format="%.3f")
    
    return summary_df

def table_2_20():
    results = json.load(open("./simulation_results/table_2_scalability_20.json", "r"))
    
    summary_df = pd.DataFrame(results)
    summary_df.set_index('Sample Size', inplace=True)
    
    # Save the summary table
    summary_df.to_latex("./tables/table_2_scalability_20.tex", float_format="%.3f")
    
    return summary_df

def table_3():
    # Load and process the data from both files
    with open("./simulation_results/table_3_tuning_full.json", "r") as f:
        results_full = pd.DataFrame(json.load(f)).T

    with open("./simulation_results/table_3_tuning_full_extended.json", "r") as f:
        results_extended = pd.DataFrame(json.load(f)).T

    # Combine the results
    all_results = pd.concat([results_full, results_extended])

    # Reset index and split it into separate columns
    all_results = all_results.reset_index()
    all_results[['model', 'observations', 'products']] = all_results['index'].str.split('|', expand=True)

    # Convert observations and products to integers
    all_results['observations'] = all_results['observations'].astype(int)
    all_results['products'] = all_results['products'].astype(int)

    # Process optimal_own data
    optimal_own = all_results.pivot(index=['model', 'observations'], 
                                    columns='products', 
                                    values='optimal_own')

    # Process optimal_cross data
    optimal_cross = all_results.pivot(index=['model', 'observations'], 
                                    columns='products', 
                                    values='optimal_cross')
    
    optimal_own.to_latex("./tables/table_3_opt_tuning_own.tex")
    optimal_own.to_csv("./tables/table_3_opt_tuning_own.csv")
    optimal_cross.to_latex("./tables/table_3_opt_tuning_cross.tex")
    
    return optimal_own, optimal_cross

def table_3_reduced_form():
    # Load and process the data from both files
    with open("./simulation_results/table_3_tuning_reduced_form.json", "r") as f:
        all_results = pd.DataFrame(json.load(f)).T

    # Reset index and split it into separate columns
    all_results = all_results.reset_index()
    all_results[['model', 'observations', 'products']] = all_results['index'].str.split('|', expand=True)

    # Convert observations and products to integers
    all_results['observations'] = all_results['observations'].astype(int)
    all_results['products'] = all_results['products'].astype(int)

    # Process optimal_own data
    optimal_own = all_results.pivot(index=['model', 'observations'], 
                                    columns='products', 
                                    values='optimal_own')
    
    optimal_own.to_latex("./tables/table_3_opt_tuning_own_reduced_form.tex")
    optimal_own.to_csv("./tables/table_3_opt_tuning_own_reduced_form.csv")
    
    return optimal_own

def table_4():
    all_results = json.load(open("./simulation_results/table_4_extra_models.json", "r"))
    
    rows = []
    for model, results in all_results.items():
        df = pd.DataFrame(results)
        for elasticity_type in ['own', 'cross']:
            true_value = df[f'true_{elasticity_type}'].mean()
            est_value = df[f'est_{elasticity_type}'].mean()
            rows.append({
                'model': model,
                'elasticity_type': elasticity_type,
                'true_value': true_value,
                'mean_estimate': est_value,
                'bias': est_value - true_value,
                'mean_se': df[f'se_{elasticity_type}'].mean(),
                'std_estimate': df[f'est_{elasticity_type}'].std()
            })
    
    summary_df = pd.DataFrame(rows)
    
    # Set 'model' and 'elasticity_type' as index
    summary_df.set_index(['model', 'elasticity_type'], inplace=True)
    
    # Save the summary table
    summary_df.to_latex("./tables/table_4_extra_models.tex", float_format="%.3f")
    
    return summary_df

def table_5():
    all_results = json.load(open("./simulation_results/table_5_supply.json", "r"))

    rows = []
    for model, results in all_results.items():
        df = pd.DataFrame(results)
        for elasticity_type in ['own', 'cross']:
            true_value = df[f'true_{elasticity_type}'].mean()
            est_value = df[f'est_{elasticity_type}'].mean()
            rows.append({
                'model': model,
                'elasticity_type': elasticity_type,
                'true_value': true_value,
                'mean_estimate': est_value,
                'bias': est_value - true_value,
                'mean_se': df[f'se_{elasticity_type}'].mean(),
                'std_estimate': df[f'est_{elasticity_type}'].std()
            })
    
    summary_df = pd.DataFrame(rows)
    
    # Set 'model' and 'elasticity_type' as index
    summary_df.set_index(['model', 'elasticity_type'], inplace=True)
    
    # Save the summary table
    summary_df.to_latex("./tables/table_5_supply.tex", float_format="%.3f")
    
    return summary_df

def table_app_pyblp():
    # Read the JSON file
    with open('./simulation_results/table_appendix_pyblp.json', 'r') as f:
        data = json.load(f)

    # Create DataFrame
    df = pd.DataFrame({
        'num_markets': data['num_markets'].values(),
        'time_1': [t[0] for t in data['time'].values()],
        'time_2': [t[1] for t in data['time'].values()]
    })

    # Calculate average time
    df['time_avg'] = (df['time_1'] + df['time_2']) / 2

    # Convert to minutes
    df['time_1'] = df['time_1'] / 60
    df['time_2'] = df['time_2'] / 60
    df['time_avg'] = df['time_avg'] / 60

    # Format numbers
    df['num_markets'] = df['num_markets'].apply(lambda x: f"{x:,}")
    df['time_1'] = df['time_1'].round(2)
    df['time_2'] = df['time_2'].round(2)
    df['time_avg'] = df['time_avg'].round(2)

    # Generate LaTeX table
    latex_table = df.to_latex(
        index=False,
        column_format='rrrr',
        float_format="%.2f",
        caption='Running Time Analysis',
        label='tab:runtime',
        header=['Number of Markets', 'Time 1 (min)', 'Time 2 (min)', 'Average Time (min)']
    )

    # Save to file
    with open('./tables/table_app_pyblp.tex', 'w') as f:
        f.write(latex_table)

    print("LaTeX table has been saved to './tables/table_app_pyblp.tex'")

def table_6():
    import numpy as np
    with open("./simulation_results/table_6_literature.json", "r") as f:
        all_results = json.load(f)

    DISPLAY_NAMES = {
        "Logit_Wollmann": "Wollmann (2018)",
        "RC_BLP": "BLP (1995)",
        "RC_Nevo": "Nevo (2001)",
    }
    MODEL_TYPES = {
        "Logit_Wollmann": "Logit",
        "RC_BLP": "RC Logit",
        "RC_Nevo": "RC Logit",
    }

    rows = []
    for config_name, r in all_results.items():
        true_elas = np.array(r['true_elas'])
        bnn_mean = np.array(r['bnn_elas_mean'])
        shares = np.array(r['mean_shares'])
        prices = np.array(r['mean_prices'])

        for j in range(len(true_elas)):
            rows.append({
                'Source': DISPLAY_NAMES.get(config_name, config_name),
                'Model': MODEL_TYPES.get(config_name, r['type']),
                'Product': j + 1,
                'True Elas.': true_elas[j],
                'BNN Elas.': bnn_mean[j],
                'Share': shares[j],
                'Price': prices[j],
            })

    df = pd.DataFrame(rows)
    df.set_index(['Source', 'Model', 'Product'], inplace=True)
    df.to_latex("./tables/table_6_literature.tex", float_format="%.3f",
                escape=True, multirow=True)
    print("Table 6 saved to ./tables/table_6_literature.tex")
    return df


if __name__ == "__main__":
    table_1()
    table_2()
    table_2_20()
    table_3()
    table_3_reduced_form()
    table_4()
    table_5()
    table_6()
    table_app_pyblp()