import matplotlib.pyplot as plt
import numpy as np
import json
from scipy import stats
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Load the min_neighbor_distance values from a JSONL file
def load_min_neighbor_distance(file_path):
    distances = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'min_neighbor_distance' in data:
                distances.append(data['min_neighbor_distance'])
    return distances

# Function to plot histogram with trendline
def plot_hist_with_trendline(ax, distances, label, color):
    """
    Plot histogram and smooth trendline on the given axes.

    Parameters:
    - ax: Axes object to plot on
    - distances: List or array of min_neighbor_distance values
    - label: Label for the dataset
    - color: Color of the bars and trendline
    """
    # Generate histogram
    hist, bins = np.histogram(distances, bins=np.linspace(0, 1, 101), density=True)
    
    # Normalize bin centers for smooth curve plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot histogram as narrow bars
    ax.bar(bin_centers, hist, width=0.01, color=color, alpha=0.7, label=label)
    
    # Add a smooth trendline using Kernel Density Estimation (KDE)
    kde = stats.gaussian_kde(distances, bw_method='scott')  # KDE with Scott's method for bandwidth
    x = np.linspace(0, 1, 1000)
    ax.plot(x, kde(x), color=color, linestyle='--', linewidth=2)

    # Set axis labels and legend
    ax.set_xlabel('Min Neighbor Distance', fontsize=16)
    ax.set_ylabel('Frequency Density', fontsize=16)
    ax.legend(loc='upper right', fontsize=14)

# Load data from JSONL files
file1 = 'results/autoif-small_distance.jsonl'
file2 = 'results/autoif-large_distance.jsonl'
distances1 = load_min_neighbor_distance(file1)
distances2 = load_min_neighbor_distance(file2)

# Create the figure and axis for plotting
fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted size for better clarity
ax.grid(linestyle="-.")

# Plot the data
plot_hist_with_trendline(ax, distances1, 'SLMs', color[0])
plot_hist_with_trendline(ax, distances2, 'LLMs', color[1])

plt.tight_layout()

# Save the plot as a PDF
plt.savefig('figs/diversity.pdf')

# Optionally, show the plot
# plt.show()
