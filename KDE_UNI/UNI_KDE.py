import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity

def compute_kde(x, data, bandwidth):
    kde = KernelDensity(kernel='tophat', bandwidth=bandwidth)
    kde.fit(data[:, None])
    log_dens = kde.score_samples(x[:, None])
    return np.exp(log_dens)

# Given centers of uniform distributions
centers = np.array([-1.553, -1.019, -0.475, -0.434, 0.395])
np.random.seed(42)  # For reproducibility

# Generate synthetic data from uniform distributions centered at given points
data = []
width = 1.7  # Width of the uniform distribution
for c in centers:
    data.append(np.random.uniform(low=c - width / 2, high=c + width / 2, size=200))  # 200 samples per distribution

# Create the plot
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Make background transparent
fig.patch.set_alpha(0)
axes[0].patch.set_alpha(0)
axes[1].patch.set_alpha(0)

# Step 1: Plot the individual uniform distributions as density plots FIRST
for dataset, center in zip(data, centers):
    sns.kdeplot(dataset, ax=axes[0], bw_adjust=1, fill=True, alpha=0.5, label=f'Center {center:.3f}')

# Step 2: Kernel Density Estimation with Uniform Kernel
x_vals = np.linspace(min(np.concatenate(data)), max(np.concatenate(data)), 500)
bandwidth = 1.0
for dataset in data:
    kde_density = compute_kde(x_vals, dataset, bandwidth)
    axes[1].plot(x_vals, kde_density, label=f'Center {np.mean(dataset):.3f}')

# Adjusting labels
axes[0].set_ylabel('')

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()