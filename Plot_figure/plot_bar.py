import matplotlib.pyplot as plt
import numpy as np

# Helper function to normalize counts and plot data
def normalize_counts(counts, total):
    return [(count / total) * 100 for count in counts]

def plot_single(ax, categories, slms_counts, llms_counts, ylabel):
    """
    Plot a single bar chart comparing SLMs and LLMs counts across categories.

    Parameters:
    - ax: The axes object to plot on
    - categories: List of categories (e.g., ['Very Easy', 'Easy', ...])
    - slms_counts: List of SLMs counts (normalized)
    - llms_counts: List of LLMs counts (normalized)
    - ylabel: Label for the y-axis (e.g., dataset name)
    """
    # Set grid lines to be dashed
    ax.grid(linestyle="-.")
    
    # Set positions for the bars
    x = np.arange(len(categories))  # Positions for categories on the x-axis
    width = 0.35  # Bar width

    # Plot the SLMs and LLMs as side-by-side bars
    ax.bar(x - width/2, slms_counts, width, label='SLMs', color='red')
    ax.bar(x + width/2, llms_counts, width, label='LLMs', color='blue')

    # Set x-axis labels and positions
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)

    # Set y-axis label
    ax.set_ylabel(ylabel, fontsize=14)
    
    # Set y-axis ticks
    ax.set_yticks(np.arange(0, max(max(slms_counts), max(llms_counts)) + 1, 20))
    
    # Display the legend
    ax.legend(loc='upper left', fontsize=12)

# Categories for the difficulty levels
categories = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']

# Data for different iterations (example data)
datasets = [
    # Alpaca Iter 1
    {"slms_counts": [42, 2201, 5177, 47, 6], "llms_counts": [21, 3933, 3517, 1, 0], "total": 7473, "label": 'Alpaca Iter1'},
    {"slms_counts": [132, 2593, 14642, 2429, 226], "llms_counts": [239, 6652, 12384, 723, 24], "total": 20022, "label": 'GSM8K Iter1'},
    {"slms_counts": [922, 3856, 30885, 12879, 3441], "llms_counts": [1055, 12782, 32988, 4740, 418], "total": 51983, "label": 'HumanEval Iter1'},
    
    # Alpaca Iter 2
    {"slms_counts": [10, 584, 6018, 778, 83], "llms_counts": [7, 2526, 4926, 14, 0], "total": 7473, "label": 'Alpaca Iter2'},
    {"slms_counts": [25, 385, 10429, 7312, 1871], "llms_counts": [20, 2050, 14554, 3214, 184], "total": 20022, "label": 'GSM8K Iter2'},
    {"slms_counts": [300, 774, 13308, 19032, 18569], "llms_counts": [193, 4098, 29721, 14274, 3697], "total": 51983, "label": 'HumanEval Iter2'},
    
    # Alpaca Iter 3
    {"slms_counts": [42, 2201, 5177, 47, 6], "llms_counts": [21, 3933, 3517, 1, 0], "total": 7473, "label": 'Alpaca Iter3'},
    {"slms_counts": [7, 61, 5165, 9098, 5691], "llms_counts": [2, 582, 11703, 6753, 982], "total": 20022, "label": 'GSM8K Iter3'},
    {"slms_counts": [77, 125, 4865, 12190, 34726], "llms_counts": [66, 1294, 19458, 19658, 11507], "total": 51983, "label": 'HumanEval Iter3'}
]

# Create the subplots (3 rows, 3 columns)
fig, axes = plt.subplots(3, 3, figsize=(15, 6))

# Plot data for each dataset
for i, data in enumerate(datasets):
    # Normalize the counts
    slms_counts = normalize_counts(data["slms_counts"], data["total"])
    llms_counts = normalize_counts(data["llms_counts"], data["total"])
    
    # Plot the data on the appropriate subplot
    row, col = divmod(i, 3)  # Determine subplot location
    plot_single(axes[row][col], categories, slms_counts, llms_counts, data["label"])

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig(f'figs/difficulty.pdf')

# Show the plot
plt.show()
