import os
import glob
from icecream import ic
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .load_json import load_json

def generate_plot(data, file_path):
    hard_instances = data['results']['hard_instances']
    plot_data = []

    # Prepare data
    for key, instance in hard_instances.items():
        if not key.startswith('iteration_'):
            continue
        try:
            iteration_num = int(key.split('_')[-1])
        except ValueError:
            continue

        iterations_val = instance.get('iterations')
        hardest_val = instance.get('hardest')
        if not isinstance(iterations_val, (int, float)) or not isinstance(hardest_val, (int, float)):
            continue

        plot_data.append({
            'iteration': iteration_num,
            'iterations': iterations_val,
            'hardest': hardest_val
        })

    if plot_data:
        df = pd.DataFrame(plot_data)
        df_melted = df.melt(id_vars='iteration', var_name='metric', value_name='value')

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_melted, x='iteration', y='value', hue='metric', marker='o')
        plt.title(f"Hardest Lital Iter. vs. Generation: {os.path.basename(file_path)}")
        plt.xlabel("Generation")
        plt.ylabel("Lital's iter")

        # Save plot
        os.makedirs('./plot/hardest_generation', exist_ok=True)
        plot_name = os.path.basename(file_path).replace('.json', '.png')
        plot_path = os.path.join('plot/hardest_generation', plot_name)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

def main():
    base_dirs = [
        "./Continuation",
        "./Results"
    ]

    total_files = 0
    error_files = 0
    warning_files = 0
    detailed_issues = []

    superplot_data = []

    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)
        
        for file_path in json_files:
            total_files += 1
            data, errors, warnings = load_json(file_path)

            # TODO: move checker to move to a error collection function
            if errors or warnings:
                entry = {
                    "file": file_path,
                    "errors": errors,
                    "warnings": warnings
                }
                detailed_issues.append(entry)
                if errors:
                    error_files += 1
                if warnings:
                    warning_files += 1

            # Generate plot only if data is valid
            if data is not None and not errors:
                generate_plot(data, file_path)

                # collect data for super plot
                hard_instances = data['results']['hard_instances']
                for key, instance in hard_instances.items():
                    if not key.startswith('iteration_'):
                        continue
                    try:
                        iteration_num = int(key.split('_')[-1])
                    except ValueError:
                        continue

                    iterations_val = instance.get('iterations')
                    hardest_val = instance.get('hardest')
                    if not isinstance(iterations_val, (int, float)) or not isinstance(hardest_val, (int, float)):
                        continue

                    superplot_data.append({
                        'filename': os.path.basename(file_path),
                        'iteration': iteration_num,
                        'metric': 'hardest',
                        'value': hardest_val
                    })

    # Print summary
    ic("=== Validation Summary ===")
    ic(total_files)
    ic(error_files)
    ic(warning_files)
    if detailed_issues:
        ic("=== Detailed Issues ===")
        for issue in detailed_issues:
            ic(issue['file'])
            if issue['errors']:
                ic("Errors:")
                for err in issue['errors']:
                    ic(err)
            if issue['warnings']:
                ic("Warnings:")
                for warn in issue['warnings']:
                    ic(warn)

    # Super plot
    if superplot_data:
        df_super = pd.DataFrame(superplot_data)

        max_iter = df_super['iteration'].max()
        filenames = df_super['filename'].unique()   

        idx = pd.MultiIndex.from_product(
            [filenames, range(1, max_iter + 1)],
            names=['filename', 'iteration']
        )

        # Reindex and forward fill missing values
        df_super = (
            df_super.set_index(['filename', 'iteration'])
            .reindex(idx)
            .reset_index()
        )
        df_super['value'] = df_super.groupby('filename')['value'].ffill()

        # Generate the plot
        plt.figure(figsize=(16, 12))
        ax = sns.lineplot(
            data=df_super,
            x='iteration',
            y='value',
            hue='filename',
            marker='.',
            #legend=False
        )
        plt.title("Hardest Lital Iterations vs. Generation")
        plt.xlabel("Generation")
        plt.ylabel("Lital's iter")
        plt.legend(title="Filename")

        # Identify the 5 highest iteration values
        top_rows_each_file = df_super.loc[df_super.groupby('filename')['value'].idxmax()]
        top5 = df_super.nlargest(5, 'value')

        for idx, row in top5.iterrows():
            x_val = row['iteration']
            y_val = row['value']
            label_text = f"{row['filename']} (val={y_val}, iter={x_val})"

            ax.annotate(
                label_text,
                xy=(x_val, y_val),
                xytext=(5, 5),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', lw=0.5),
                fontsize=8
            )

        # Save the super plot
        super_plot_path = os.path.join('plot/hardest_generation', 'all_hardest_vs_generation.png')
        plt.savefig(super_plot_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
