import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Define parameters
mean = 10  # Mean of the lognormal distribution in normal space
sigma_values = [0.25, 0.5, 1.0, 2.0]  # Different sigma values (shape parameter)

# Generate x values (range of the PDF)
x = np.linspace(0, 40, 1000)  # Adjust range as needed

# Plot PDF for each sigma
plt.figure(figsize=(8, 6))  # Set figure size

for sigma in sigma_values:
    # Calculate the log-space mean mu
    mu = np.log(mean) - (sigma ** 2) / 2
    scale = np.exp(mu)
    
    # Generate the PDF using lognorm from scipy
    pdf = lognorm.pdf(x, s=sigma, scale=scale)
    
    # Plot using Seaborn
    sns.lineplot(x=x, y=pdf, label=f'sigma = {sigma}', linewidth=2)

# Customize the plot
plt.title('Lognormal Distribution Probability Density Function')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()