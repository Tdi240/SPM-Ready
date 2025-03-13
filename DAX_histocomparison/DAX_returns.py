import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.stats import t, norm

# Step 1: Fetch DAX Data from Yahoo Finance
dax = yf.download('^GDAXI', start='2000-01-01', end='2024-01-01', auto_adjust=False)  # Fetch historical data
dax['Returns'] = dax['Close'].pct_change().dropna()  # Compute daily returns
returns = dax['Returns'].dropna().values  # Convert to NumPy array

# Step 2: Create the Transparent Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Make background transparent
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# Histogram of returns
sns.histplot(returns, bins=50, color='lightblue', kde=False, stat='density', label='Histogram', ax=ax)

# Define range for distribution plots
x_vals = np.linspace(returns.min(), returns.max(), 1000)

# Epanechnikov KDE
bandwidth = 0.005  # Adjust bandwidth for smooth KDE
kde_epanechnikov = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(returns[:, np.newaxis])
log_dens_epanechnikov = kde_epanechnikov.score_samples(x_vals[:, np.newaxis])
ax.plot(x_vals, np.exp(log_dens_epanechnikov), color='red', lw=2, label='Epanechnikov KDE')

# Normal Distribution Fit
mu, sigma = np.mean(returns), np.std(returns)
normal_dist = norm.pdf(x_vals, mu, sigma)
ax.plot(x_vals, normal_dist, color='blue', lw=2, label='Normal')

# Student's t Distribution (df=5)
df = 5
student_t_dist = t.pdf(x_vals, df=df, loc=mu, scale=sigma)
ax.plot(x_vals, student_t_dist, color='green', lw=2, label="Student's t, 5 df")

# Customize plot
ax.set_xlabel('Daily Returns')
ax.set_ylabel('Density')
ax.legend(fancybox=True, framealpha=0.5)
plt.tight_layout()

# Show the plot
plt.show()
