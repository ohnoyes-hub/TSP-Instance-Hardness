import numpy as np
from scipy import stats
from util.load_experiment import load_all_hard_instances
import pandas as pd
def analyze_search_trajectories(df_hard_instances):
    """
    Analyze how search progresses through generations without LON.
    """
    
    trajectories = {}
    
    # Group by experiment configuration
    config_cols = ['mutation_type', 'generation_type', 'distribution', 'city_size', 'range']
    
    for config, group in df_hard_instances.groupby(config_cols):
        # Sort by generation
        trajectory = group.sort_values('generation')
        
        # Calculate trajectory metrics
        hardness_values = trajectory['hardest_value'].values
        generations = trajectory['generation'].values
        
        # Trajectory characteristics
        metrics = {
            'config': dict(zip(config_cols, config)),
            'final_hardness': hardness_values[-1] if len(hardness_values) > 0 else 0,
            'hardness_improvement': hardness_values[-1] - hardness_values[0] if len(hardness_values) > 1 else 0,
            'trajectory_length': len(hardness_values),
            'plateau_ratio': calculate_plateau_ratio(hardness_values),
            'improvement_rate': calculate_improvement_rate(generations, hardness_values),
            'volatility': np.std(np.diff(hardness_values)) if len(hardness_values) > 1 else 0
        }
        
        trajectories[config] = metrics
    
    return pd.DataFrame.from_dict(trajectories, orient='index')

def calculate_plateau_ratio(hardness_values, threshold=0.01):
    """Calculate proportion of trajectory spent in plateaus."""
    if len(hardness_values) < 2:
        return 0
    
    differences = np.diff(hardness_values)
    plateau_steps = np.sum(np.abs(differences) < threshold)
    return plateau_steps / len(differences)

def calculate_improvement_rate(generations, hardness_values):
    """Calculate rate of hardness improvement."""
    if len(generations) < 2:
        return 0
    
    # Fit linear regression
    slope, _, r_value, _, _ = stats.linregress(generations, hardness_values)
    return {'slope': slope, 'r_squared': r_value**2}

if __name__ == "__main__":
    df_hard_instances = load_all_hard_instances()
    df_trajectories = analyze_search_trajectories(df_hard_instances)
    df_trajectories.to_csv('search_trajectories_analysis.csv', index=False)
    print("Search trajectory analysis saved to 'search_trajectories_analysis.csv'")

