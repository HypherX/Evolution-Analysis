import matplotlib.pyplot as plt
import numpy as np

color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_single(ax, x, model1, Ours, group_labels, y_min, y_max, ylabel):
    """
    Plot a single line graph comparing two models.

    Parameters:
    - ax: The axes object to plot on.
    - x: x-axis data (e.g., iterations).
    - model1: List or array of values for the first model (e.g., LLMs).
    - Ours: List or array of values for the second model (e.g., SLMs).
    - group_labels: List of labels for each x-tick.
    - y_ticks: List of y-axis tick values.
    - ylim: Tuple representing the y-axis limits (min, max).
    - ylabel: Label for the y-axis.
    """
    # Set dashed grid lines
    ax.grid(linestyle="-.")
    
    # Plot both models with distinct markers and colors
    ax.plot(x, model1, marker='s', markersize=5, color=color[1], label="LLMs", linewidth=2.0)
    ax.plot(x, Ours, marker='X', markersize=5, color=color[0], label="SLMs", linewidth=2.0)

    # Set x-axis labels and ticks
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=14)
    
    
    # Set y-axis range
    ax.set_ylim([y_min, y_max])
    ax.set_yticks(np.linspace(y_min, y_max, 5))
    ax.set_yticklabels([str(int(tick)) for tick in ax.get_yticks()], fontsize=14)
    
    # Set y-axis label
    ax.set_ylabel(ylabel, fontsize=16)
    
    # Display legend
    ax.legend(loc='upper left', numpoints=1, ncol=2, fontsize=16)

# X-axis values (iterations)
x = np.array([1, 2, 3, 4])

# Labels for each iteration
group_labels = ['Iter-0', 'Iter-1', 'Iter-2', 'Iter-3']

# Create the subplots: 2 rows, 4 columns
fig, axes = plt.subplots(2, 4, figsize=(18, 6))

# Data for each plot (model1 and Ours values)
# plot_data = [
#     {"model1": [32.53, 31.61, 29.94, 38.82], "Ours": [32.53, 38.63, 43.99, 36.41], "y_ticks": [29, 34, 39, 44], "ylim": (29, 45), "ylabel": "Pr.(S) (%)"},
#     {"model1": [44.24, 43.29, 41.85, 50.60], "Ours": [44.24, 48.56, 53.36, 45.80], "y_ticks": [41, 45.5, 50, 54.5], "ylim": (41, 54.5), "ylabel": "In.(S) (%)"},
#     {"model1": [35.30, 34.38, 32.72, 42.70], "Ours": [35.30, 43.33, 46.58, 40.85], "y_ticks": [32, 37, 42, 47], "ylim": (32, 47), "ylabel": "Pr.(L) (%)"},
#     {"model1": [46.88, 46.16, 45.20, 53.96], "Ours": [46.88, 51.92, 56.00, 49.76], "y_ticks": [45, 49, 53, 57], "ylim": (44.5, 57), "ylabel": "In.(L) (%)"},
#     {"model1": [63.15, 63.23, 67.17, 65.50], "Ours": [63.15, 63.91, 68.46, 67.85], "y_ticks": [63, 65, 67, 69], "ylim": (62.5, 69), "ylabel": "GSM8K (%)"},
#     {"model1": [6.94, 9.22, 11.70, 10.30], "Ours": [6.94, 11.48, 12.80, 13.42], "y_ticks": [6, 8.5, 11, 13.5], "ylim": (6, 13.5), "ylabel": "MATH (%)"},
#     {"model1": [41.46, 45.73, 49.39, 47.56], "Ours": [41.46, 48.17, 50.00, 43.29], "y_ticks": [40, 43.5, 47, 50.5], "ylim": (40, 50.5), "ylabel": "HumanEval (%)"},
#     {"model1": [33.00, 37.80, 40.60, 40.80], "Ours": [33.00, 39.40, 40.40, 40.60], "y_ticks": [32, 35, 38, 41], "ylim": (32, 41), "ylabel": "MBPP (%)"}
# ]

plot_data = [
    {"model1": [23.11, 31.61, 29.94, 35.12], "Ours": [23.11, 38.63, 43.99, 33.09], "ylabel": "Pr.(S)"},
    {"model1": [32.97, 43.29, 41.85, 47.36], "Ours": [32.97, 48.56, 53.36, 44.72], "ylabel": "In.(S)"},
    {"model1": [24.77, 34.38, 32.72, 36.97], "Ours": [24.77, 43.33, 46.58, 36.41], "ylabel": "Pr.(L)"},
    {"model1": [35.13, 46.16, 45.20, 49.28], "Ours": [35.13, 51.92, 56.00, 48.32], "ylabel": "In.(L)"},
    {"model1": [53.68, 63.23, 67.17, 65.50], "Ours": [53.68, 63.91, 68.46, 67.85], "ylabel": "GSM8K"},
    {"model1": [0.22, 9.22, 11.70, 10.30], "Ours": [0.22, 11.48, 12.80, 13.42], "ylabel": "MATH"},
    {"model1": [25.00, 45.73, 49.39, 47.56], "Ours": [25.00, 48.17, 50.00, 43.29], "ylabel": "HumanEval"},
    {"model1": [28.60, 37.80, 40.60, 40.80], "Ours": [28.60, 39.40, 40.40, 40.60], "ylabel": "MBPP"}
]

# Loop through the data and plot each subplot
for i, data in enumerate(plot_data):
    data["model1"] = np.array(data["model1"])
    data["Ours"] = np.array(data["Ours"])
    row, col = divmod(i, 4)  # Determine row and column for subplot
    y_min = min(int(data["model1"].min()), int(data["Ours"].min())) - 1
    y_max = max(int(data["model1"].max()), int(data["Ours"].max())) + 2
    plot_single(
        axes[row, col], x, np.array(data["model1"]), np.array(data["Ours"]), 
        group_labels, y_min, y_max, data["ylabel"]
    )

# Adjust layout to avoid overlap between subplots
plt.tight_layout()

# Save the figure as a PDF file
plt.savefig(f'figs/iterations.pdf')
plt.savefig(f'figs/iterations.png')

# Display the plot
# plt.show()
