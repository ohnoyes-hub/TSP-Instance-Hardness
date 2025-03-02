# Add these imports at the top with other imports
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from icecream import ic
from .load_json import load_json

def visualize_iteration_counts(directory_path):
   
    all_iterations = []
    file_count = 0
    
    for json_file in glob.glob(directory_path):
        file_count += 1
        data, errors, warnings = load_json(json_file)
        
        if data is None:
            ic(json_file, errors)
            continue
            
        results = data.get('results', {})
        iterations = results.get('all_iterations', [])
        
        # Additional type check (already validated, but safe)
        if isinstance(iterations, list) and all(isinstance(i, int) for i in iterations):
            all_iterations.extend(iterations)
        else:
            print(f"⚠️ Invalid iterations in {json_file}, type: {type(iterations)}")
            continue
    
    if not all_iterations:
        ic("No valid iteration data found across all files")
        return
    
    print(f"Visualizing data from {file_count} files ({len(all_iterations)} total iterations)")
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Create ordered count plot
    count_data = sorted(Counter(all_iterations).items())
    ax = sns.barplot(
        x=[k for k, v in count_data],
        y=[v for k, v in count_data],
        palette="viridis"
    )
    
    # Formatting
    ax.set_title('Iteration Frequency Across All Experiments', pad=20)
    ax.set_xlabel('Iteration Number', labelpad=10)
    ax.set_ylabel('Occurrence Count', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_iteration_counts('Results/**/*.json')