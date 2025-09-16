import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import your loader module
from util.load_experiment import load_all_matrices, load_initial_and_hard_instances

def calculate_matrix_features(matrix: np.ndarray) -> dict:
    """
    Calculate comprehensive statistical features from a TSP distance matrix.
    
    Parameters:
    -----------
    matrix : np.ndarray
        The distance matrix
    
    Returns:
    --------
    dict : Dictionary of calculated features
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # Extract upper triangle (excluding diagonal)
    upper_triangle_indices = np.triu_indices(matrix.shape[0], k=1)
    distances = matrix[upper_triangle_indices]
    
    # Filter out infinite values
    distances = distances[np.isfinite(distances)]
    
    if len(distances) == 0:
        return {f: np.nan for f in ['std', 'mean', 'cv', 'min', 'max', 'range', 
                                    'skew', 'kurtosis', 'q25', 'q75', 'iqr', 'median']}
    
    features = {
        'std': np.std(distances),
        'mean': np.mean(distances),
        'cv': np.std(distances) / np.mean(distances) if np.mean(distances) != 0 else np.nan, # coefficient of variation
        'min': np.min(distances),
        'max': np.max(distances),
        'range': np.max(distances) - np.min(distances),
        'skew': stats.skew(distances),
        'kurtosis': stats.kurtosis(distances),
        'q25': np.percentile(distances, 25),
        'q75': np.percentile(distances, 75),
        'iqr': np.percentile(distances, 75) - np.percentile(distances, 25),
        'median': np.median(distances),
        'mad': np.median(np.abs(distances - np.median(distances))),  # Median absolute deviation
        'entropy': stats.entropy(np.histogram(distances, bins=20)[0] + 1e-10),  # Distribution entropy
    }
    
    # Add relative features
    features['range_ratio'] = features['range'] / features['mean'] if features['mean'] != 0 else np.nan
    features['iqr_ratio'] = features['iqr'] / features['median'] if features['median'] != 0 else np.nan
    
    return features

def merge_with_iterations(df_matrices: pd.DataFrame) -> pd.DataFrame:
    """
    Merge matrix features with iteration data (Lital iterations).
    
    Parameters:
    -----------
    df_matrices : pd.DataFrame
        DataFrame from load_all_matrices()
    
    Returns:
    --------
    pd.DataFrame : DataFrame with features and iterations
    """
    print("Calculating matrix features...")
    
    # Calculate all features for each matrix
    feature_dicts = df_matrices['matrix'].apply(calculate_matrix_features)
    features_df = pd.DataFrame(list(feature_dicts))
    
    # Add prefix to feature columns
    features_df.columns = ['dist_' + col for col in features_df.columns]
    
    # Combine with original dataframe
    df_combined = pd.concat([df_matrices.reset_index(drop=True), features_df], axis=1)
    
    # Filter to only rows with iteration data
    df_with_iterations = df_combined[df_combined['iteration'].notna()].copy()
    
    print(f"Found {len(df_with_iterations)} matrices with iteration data")
    
    return df_with_iterations

def analyze_hardness_correlation(df: pd.DataFrame) -> dict:
    """
    Analyze correlation between distance features and instance hardness (iterations).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and iterations
    
    Returns:
    --------
    dict : Correlation analysis results
    """
    # Get feature columns
    feature_cols = [col for col in df.columns if col.startswith('dist_')]
    
    # Remove rows with NaN in iterations or features
    df_clean = df[['iteration'] + feature_cols].dropna()
    
    if len(df_clean) == 0:
        print("No valid data for correlation analysis")
        return {}
    
    # Calculate correlations
    correlations = {}
    for col in feature_cols:
        if df_clean[col].std() > 0:  # Only if feature has variation
            corr, p_value = stats.pearsonr(df_clean[col], df_clean['iteration'])
            correlations[col] = {
                'pearson_r': corr,
                'p_value': p_value,
                'spearman_r': stats.spearmanr(df_clean[col], df_clean['iteration'])[0],
                'significant': p_value < 0.05
            }
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), 
                        key=lambda x: abs(x[1]['pearson_r']), 
                        reverse=True)
    
    return dict(sorted_corr)

def plot_hardness_correlations(df: pd.DataFrame, top_n: int = 6, save_path: str = None):
    """
    Create visualizations of correlations between features and hardness.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and iterations
    top_n : int
        Number of top correlated features to plot
    save_path : str
        Optional path to save the figure
    """
    # Get feature columns
    feature_cols = [col for col in df.columns if col.startswith('dist_')]
    df_clean = df[['iteration'] + feature_cols].dropna()
    
    if len(df_clean) == 0:
        print("No valid data for plotting")
        return
    
    # Calculate correlations to find top features
    correlations = {}
    for col in feature_cols:
        if df_clean[col].std() > 0:
            corr = abs(stats.pearsonr(df_clean[col], df_clean['iteration'])[0])
            correlations[col] = corr
    
    top_features = sorted(correlations.keys(), 
                         key=lambda x: correlations[x], 
                         reverse=True)[:top_n]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Instance Hardness vs Distance Features', fontsize=16)
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Scatter plot with regression line
        x = df_clean[feature]
        y = df_clean['iteration']
        
        ax.scatter(x, y, alpha=0.5, s=20)
        
        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2)
        
        # Add correlation info
        corr = stats.pearsonr(x, y)[0]
        ax.set_xlabel(feature.replace('dist_', ''))
        ax.set_ylabel('Iterations (Hardness)')
        ax.set_title(f'{feature.replace("dist_", "").upper()}\nPearson r = {corr:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def create_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """
    Create a comprehensive correlation heatmap including iterations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and iterations
    save_path : str
        Optional path to save the figure
    """
    # Select relevant columns
    feature_cols = [col for col in df.columns if col.startswith('dist_')]
    analysis_cols = ['iteration'] + feature_cols
    
    # Add other numerical parameters if available
    for col in ['city_size', 'range', 'generation']:
        if col in df.columns:
            analysis_cols.append(col)
    
    df_clean = df[analysis_cols].dropna()
    
    if len(df_clean) > 1:
        # Calculate correlation matrix
        corr_matrix = df_clean.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1)
        
        plt.title('Correlation Matrix: Instance Hardness and Features', fontsize=14)
        
        # Highlight iteration row/column
        ax = plt.gca()
        iteration_idx = list(corr_matrix.columns).index('iteration')
        ax.add_patch(plt.Rectangle((0, iteration_idx), len(corr_matrix.columns), 1,
                                  fill=False, edgecolor='green', lw=2))
        ax.add_patch(plt.Rectangle((iteration_idx, 0), 1, len(corr_matrix.columns),
                                  fill=False, edgecolor='green', lw=2))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        plt.show()

def build_hardness_predictor(df: pd.DataFrame) -> dict:
    """
    Build predictive models for instance hardness based on distance features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and iterations
    
    Returns:
    --------
    dict : Model performance metrics and feature importance
    """
    # Prepare data
    feature_cols = [col for col in df.columns if col.startswith('dist_')]
    df_clean = df[['iteration'] + feature_cols].dropna()
    
    if len(df_clean) < 10:
        print("Insufficient data for modeling")
        return {}
    
    X = df_clean[feature_cols]
    y = np.log1p(df_clean['iteration'])  # Log transform for better prediction
    
    # Remove features with no variation
    valid_features = [col for col in X.columns if X[col].std() > 0]
    X = X[valid_features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring='r2')
    lr.fit(X_scaled, y)
    results['linear_regression'] = {
        'r2_mean': lr_scores.mean(),
        'r2_std': lr_scores.std(),
        'coefficients': dict(zip(valid_features, lr.coef_))
    }
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge_scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring='r2')
    ridge.fit(X_scaled, y)
    results['ridge_regression'] = {
        'r2_mean': ridge_scores.mean(),
        'r2_std': ridge_scores.std(),
        'coefficients': dict(zip(valid_features, ridge.coef_))
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    rf.fit(X, y)
    results['random_forest'] = {
        'r2_mean': rf_scores.mean(),
        'r2_std': rf_scores.std(),
        'feature_importance': dict(zip(valid_features, rf.feature_importances_))
    }
    
    return results

def plot_feature_importance(model_results: dict, save_path: str = None):
    """
    Plot feature importance from different models.
    
    Parameters:
    -----------
    model_results : dict
        Results from build_hardness_predictor
    save_path : str
        Optional path to save the figure
    """
    if not model_results:
        print("No model results to plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Feature Importance for Hardness Prediction', fontsize=16)
    
    # Linear Regression coefficients
    if 'linear_regression' in model_results:
        ax = axes[0]
        coef = model_results['linear_regression']['coefficients']
        sorted_coef = sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        features, values = zip(*sorted_coef)
        features = [f.replace('dist_', '') for f in features]
        
        ax.barh(features, values)
        ax.set_xlabel('Coefficient Value')
        ax.set_title(f'Linear Regression\nR² = {model_results["linear_regression"]["r2_mean"]:.3f}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Ridge Regression coefficients
    if 'ridge_regression' in model_results:
        ax = axes[1]
        coef = model_results['ridge_regression']['coefficients']
        sorted_coef = sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        features, values = zip(*sorted_coef)
        features = [f.replace('dist_', '') for f in features]
        
        ax.barh(features, values)
        ax.set_xlabel('Coefficient Value')
        ax.set_title(f'Ridge Regression\nR² = {model_results["ridge_regression"]["r2_mean"]:.3f}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Random Forest importance
    if 'random_forest' in model_results:
        ax = axes[2]
        importance = model_results['random_forest']['feature_importance']
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        features, values = zip(*sorted_imp)
        features = [f.replace('dist_', '') for f in features]
        
        ax.barh(features, values)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Random Forest\nR² = {model_results["random_forest"]["r2_mean"]:.3f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def analyze_hardness_by_configuration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how hardness correlates with features across different configurations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and iterations
    
    Returns:
    --------
    pd.DataFrame : Summary statistics by configuration
    """
    config_params = ['distribution', 'generation_type', 'mutation_type', 'city_size']
    available_params = [p for p in config_params if p in df.columns]
    
    summary_data = []
    
    for param in available_params:
        for value in df[param].unique():
            if pd.isna(value):
                continue
                
            subset = df[df[param] == value]
            
            if len(subset) < 5:  # Skip small groups
                continue
            
            # Calculate correlation between std and iterations
            if 'dist_std' in subset.columns:
                valid_data = subset[['dist_std', 'iteration']].dropna()
                if len(valid_data) > 3:
                    corr, p_val = stats.pearsonr(valid_data['dist_std'], 
                                                 valid_data['iteration'])
                else:
                    corr, p_val = np.nan, np.nan
            else:
                corr, p_val = np.nan, np.nan
            
            summary_data.append({
                'parameter': param,
                'value': value,
                'n_samples': len(subset),
                'mean_iteration': subset['iteration'].mean(),
                'std_iteration': subset['iteration'].std(),
                'mean_dist_std': subset['dist_std'].mean() if 'dist_std' in subset.columns else np.nan,
                'correlation': corr,
                'p_value': p_val,
                'significant': p_val < 0.05 if not np.isnan(p_val) else False
            })
    
    return pd.DataFrame(summary_data)

def main():
    """
    Main function to run the complete hardness correlation analysis.
    """
    print("="*60)
    print("TSP INSTANCE HARDNESS CORRELATION ANALYSIS")
    print("="*60)
    
    # Load data
    print("\nLoading matrix data...")
    df_matrices = load_all_matrices()
    print(f"Loaded {len(df_matrices)} matrices")
    
    # Merge with iterations and calculate features
    print("\nMerging with iteration data and calculating features...")
    df_analyzed = merge_with_iterations(df_matrices)
    
    if len(df_analyzed) == 0:
        print("No data with iterations found. Cannot perform hardness analysis.")
        return None, None
    
    # Analyze correlations
    print("\nAnalyzing correlations with hardness...")
    correlations = analyze_hardness_correlation(df_analyzed)
    
    # Print top correlations
    print("\n" + "="*60)
    print("TOP FEATURES CORRELATED WITH INSTANCE HARDNESS")
    print("="*60)
    print(f"{'Feature':<20} {'Pearson r':<12} {'Spearman r':<12} {'P-value':<12} {'Significant':<12}")
    print("-"*60)
    
    for feature, stats in list(correlations.items())[:10]:
        feature_name = feature.replace('dist_', '')
        print(f"{feature_name:<20} {stats['pearson_r']:>11.3f} {stats['spearman_r']:>11.3f} "
              f"{stats['p_value']:>11.4f} {'Yes' if stats['significant'] else 'No':>11}")
    
    # Build predictive models
    print("\n" + "="*60)
    print("BUILDING PREDICTIVE MODELS")
    print("="*60)
    model_results = build_hardness_predictor(df_analyzed)
    
    for model_name, results in model_results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Cross-validated R² = {results['r2_mean']:.3f} (±{results['r2_std']:.3f})")
        
        if model_name == 'random_forest':
            print("  Top 5 Important Features:")
            sorted_importance = sorted(results['feature_importance'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]
            for feat, imp in sorted_importance:
                print(f"    {feat.replace('dist_', '')}: {imp:.3f}")
    
    # Analyze by configuration
    print("\n" + "="*60)
    print("HARDNESS CORRELATION BY CONFIGURATION")
    print("="*60)
    config_summary = analyze_hardness_by_configuration(df_analyzed)
    
    if not config_summary.empty:
        # Show configurations with strongest correlations
        config_summary_sorted = config_summary.sort_values('correlation', 
                                                          key=lambda x: abs(x), 
                                                          ascending=False)
        print("\nTop configurations with strongest feature-hardness correlation:")
        print(config_summary_sorted[['parameter', 'value', 'n_samples', 
                                    'mean_iteration', 'correlation', 'significant']].head(10))
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_hardness_correlations(df_analyzed, save_path='hardness_correlations.png')
    create_correlation_heatmap(df_analyzed, save_path='hardness_heatmap.png')
    plot_feature_importance(model_results, save_path='feature_importance.png')
    
    # Save results
    output_file = 'hardness_correlation_analysis.csv'
    df_analyzed.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")
    
    # Save correlation summary
    corr_df = pd.DataFrame(correlations).T
    corr_df.to_csv('feature_hardness_correlations.csv')
    print(f"Correlation summary saved to feature_hardness_correlations.csv")
    
    return df_analyzed, correlations

if __name__ == "__main__":
    df_results, correlation_results = main()
    
    # Additional analysis: Does variation predict hardness?
    if df_results is not None and 'dist_cv' in df_results.columns:
        print("\n" + "="*60)
        print("COEFFICIENT OF VARIATION AS HARDNESS PREDICTOR")
        print("="*60)
        
        valid_data = df_results[['dist_cv', 'iteration']].dropna()
        if len(valid_data) > 0:
            corr, p_val = stats.pearsonr(valid_data['dist_cv'], valid_data['iteration'])
            print(f"Correlation between CV and hardness: {corr:.3f} (p={p_val:.4f})")
            
            # Simple threshold analysis
            median_cv = valid_data['dist_cv'].median()
            high_cv = valid_data[valid_data['dist_cv'] > median_cv]['iteration'].mean()
            low_cv = valid_data[valid_data['dist_cv'] <= median_cv]['iteration'].mean()
            print(f"\nAverage iterations for high CV instances: {high_cv:.0f}")
            print(f"Average iterations for low CV instances: {low_cv:.0f}")
            print(f"Ratio: {high_cv/low_cv:.2f}x harder for high variation instances")