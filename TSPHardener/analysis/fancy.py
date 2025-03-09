import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform

def plot_uniform_histogram(rand_max, num_samples=1000, num_bins=20):
    data = np.random.uniform(0, rand_max, num_samples)
    
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(8, 6))
    
    sns.histplot(data, bins=num_bins, kde=False, stat="density", color="skyblue", edgecolor="black")
    
    x = np.linspace(0, rand_max, 100)
    pdf = uniform.pdf(x, loc=0, scale=rand_max)
    plt.plot(x, pdf, 'r-', lw=2, label='Uniform PDF')
    
    plt.xlim(0, rand_max)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Uniform Distribution Histogram on [0, rand_max = {rand_max}]", fontsize=14)
    plt.legend()
    plt.savefig("./plot/uniform-drawn-method.png")

plot_uniform_histogram(rand_max=5, num_samples=5000, num_bins=25)

from scipy.stats import lognorm

def plot_lognormal_histograms(std_devs, num_samples=1000, num_bins=20):
    mean = 0
    colors = sns.color_palette("husl", len(std_devs))
    
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(8, 6))
    
    for sigma, color in zip(std_devs, colors):
        data = np.random.lognormal(mean, sigma, num_samples)
        
        # Plot histogram with Seaborn
        sns.histplot(data, bins=num_bins, kde=False, stat="density", color=color, edgecolor="black", alpha=0.5, label=f"σ = {sigma}")
        
        # Plot the theoretical PDF
        x = np.linspace(min(data), max(data), 1000)
        pdf = lognorm.pdf(x, sigma, scale=np.exp(mean))
        plt.plot(x, pdf, '-', lw=2, color=color)
    
    plt.xlabel("x", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Lognormal Distributions with Different σ and μ = 0", fontsize=14)
    plt.legend()
    plt.savefig("./plot/lognormal-drawn-method.png")

plot_lognormal_histograms(std_devs=[0.2,0.4,0.8], num_samples=5000, num_bins=25)
