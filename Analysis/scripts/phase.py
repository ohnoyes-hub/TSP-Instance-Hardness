import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

from util.load_experiment import load_phase_transition_iterations

def plot_phase_transition_city30_atsp_uniform():
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="viridis",
        rc={
            "figure.figsize": (10, 6),
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "font.weight": "bold",
        }
    )
    
    df = load_phase_transition_iterations()
    
    df = df[
        (df['distribution'] == 'uniform') &
        (df['generation_type'] == 'asymmetric') &
        (df['city_size'] == 30) &
        (df['iteration'] > 0)
    ].copy()
    
    if df.empty:
        print("No data found for specified settings!")
        return

    stats = df.groupby('range')['iteration'].agg(['median', 'mean', 'std']).reset_index()
    stats = stats.sort_values('range')
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.fill_between(
        stats['range'],
        stats['median'] - 0.5 * stats['std'],
        stats['median'] + 0.5 * stats['std'],
        color='gray', alpha=0.2, label='±0.5 Std Dev'
    )
    # Median and mean lines
    sns.lineplot(x='range', y='median', data=stats, marker="s", linewidth=2, label='Median')
    sns.lineplot(x='range', y='mean', data=stats, marker="^", linewidth=2, label='Mean')
    # Scatter of all raw points
    sns.scatterplot(x='range', y='iteration', data=df, color='k', alpha=0.2, s=18, label='Raw')
    
    plt.xlabel(r"$rand_{max}$")
    plt.ylabel("Lital Iteration")
    # text inset
    inset_test = 'ATSP\n30-City\nUniform'
    plt.text(0.05, 0.95, inset_test,
             ha='left', va='top',
             transform=plt.gca().transAxes,
             fontsize=14, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.7),
             zorder=10)
    """
    inset_text = f"{tsp_type[0].capitalize()}TSP\n{size}-City\n{dist.capitalize()}"
    ax.text(
        0.05, 0.95, inset_text,
        ha='left', va='top',
        transform=ax.transAxes,
        fontsize=14, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.7),
        zorder=10
    )
    """
    
    plt.legend()
    plt.tight_layout()
    

    os.makedirs('./plot/phase_transition', exist_ok=True)
    plt.savefig('./plot/phase_transition/phase_transition_city30_atsp_uniform.png', dpi=300)
    plt.close()
    print("Plot saved to ./plot/phase_transition/phase_transition_city30_atsp_uniform.png")

def plot_phase_transition_city30_stsp_uniform():
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="viridis",
        rc={
            "figure.figsize": (10, 6),
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "font.weight": "bold",
        }
    )
    
    df = load_phase_transition_iterations()
    
    df = df[
        (df['distribution'] == 'uniform') &
        (df['generation_type'] == 'symmetric') &
        (df['city_size'] == 30) &
        (df['iteration'] > 0)
    ].copy()
    
    if df.empty:
        print("No data found for specified settings!")
        return

    stats = df.groupby('range')['iteration'].agg(['median', 'mean', 'std']).reset_index()
    stats = stats.sort_values('range')
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.fill_between(
        stats['range'],
        stats['median'] - 0.5 * stats['std'],
        stats['median'] + 0.5 * stats['std'],
        color='gray', alpha=0.2, label='±0.5 Std Dev'
    )
    # Median and mean lines
    sns.lineplot(x='range', y='median', data=stats, marker="s", linewidth=2, label='Median')
    sns.lineplot(x='range', y='mean', data=stats, marker="^", linewidth=2, label='Mean')
    # Scatter of all raw points
    sns.scatterplot(x='range', y='iteration', data=df, color='k', alpha=0.2, s=18, label='Raw')
    
    plt.xlabel(r"$rand_{max}$")
    plt.ylabel("Lital Iteration")
    # text inset
    inset_test = 'ATSP\n30-City\nUniform'
    plt.text(0.05, 0.95, inset_test,
             ha='left', va='top',
             transform=plt.gca().transAxes,
             fontsize=14, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.7),
             zorder=10)
    """
    inset_text = f"{tsp_type[0].capitalize()}TSP\n{size}-City\n{dist.capitalize()}"
    ax.text(
        0.05, 0.95, inset_text,
        ha='left', va='top',
        transform=ax.transAxes,
        fontsize=14, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.7),
        zorder=10
    )
    """
    
    plt.legend()
    plt.tight_layout()
    

    os.makedirs('./plot/phase_transition', exist_ok=True)
    plt.savefig('./plot/phase_transition/phase_transition_city30_stsp_uniform.png', dpi=300)
    plt.close()
    print("Plot saved to ./plot/phase_transition/phase_transition_city30_stsp_uniform.png")

if __name__ == "__main__":
    plot_phase_transition_city30_atsp_uniform()
    plot_phase_transition_city30_stsp_uniform()