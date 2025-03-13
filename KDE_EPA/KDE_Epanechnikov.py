import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity

def compute_kde(x, data, bandwidth):
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth)
    kde.fit(data[:, None])
    log_dens = kde.score_samples(x[:, None])
    return np.exp(log_dens)

# Given centers of distributions
centers = np.array([-1.553, -1.019, -0.475, -0.434, 0.395])
np.random.seed(42)  # For reproducibility

# Generate synthetic data from an Epanechnikov-like distribution
data = []
width = 1.6 # spread of the distribution
for c in centers:
    u = np.random.uniform(-1, 1, 200)  # Generate 200 samples
    epanechnikov_samples = c + width * u * np.sqrt(1 - u**2)  # Transform to Epanechnikov shape
    data.append(epanechnikov_samples)

# Create the plot
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Make background transparent
fig.patch.set_alpha(0)
axes[0].patch.set_alpha(0)
axes[1].patch.set_alpha(0)

for dataset, center in zip(data, centers):
    sns.kdeplot(dataset, ax=axes[0], bw_adjust=1, fill=True, alpha=0.5, label=f'Center {center:.3f}')

x_vals = np.linspace(min(np.concatenate(data)), max(np.concatenate(data)), 500)
bandwidth = 1.4
for dataset in data:
    kde_density = compute_kde(x_vals, dataset, bandwidth)
    axes[1].plot(x_vals, kde_density, label=f'Center {np.mean(dataset):.3f}')

# Adjusting labels
axes[0].set_ylabel('')

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()