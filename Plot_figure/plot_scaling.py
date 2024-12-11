import matplotlib.pyplot as plt
import numpy as np

# 定义颜色列表
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 定义绘图函数
def plot_single(ax, x, y1, y2, xlabels, ylabel, color1, color2):
    ax.grid(linestyle="-.")

    ax.plot(x, y1, marker='s', markersize=5, color=color1, label='LLMs', linewidth=2.0)
    ax.plot(x, y2, marker='X', markersize=5, color=color2, label='SLMs', linewidth=2.0)
    
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=14)
    
    # 设置y轴标签和刻度
    y_min = min(int(y1.min()), int(y2.min())) - 4
    y_max = max(int(y1.max()), int(y2.max())) + 4
    ax.set_ylim([y_min, y_max])
    ax.set_yticks(np.linspace(y_min, y_max, 5))
    ax.set_yticklabels([str(int(tick)) for tick in ax.get_yticks()], fontsize=14)
    
    # 设置y轴标签
    ax.set_ylabel(ylabel, fontsize=16)
    
    # 显示图例
    ax.legend(loc='upper left', numpoints=1, ncol=2, fontsize=12)

# X轴值（不同的B值）
x = np.array(['0.5B', '1.5B', '3B', '7B', '14B', '32B', '72B'])

# 创建一行三列的子图
fig, axes = plt.subplots(2, 4, figsize=(18, 6))

# 数据（这里需要您提供具体的数据）
# 示例数据，您需要替换成实际的数据
PrS_llms = np.array([18.48, 28.84, 37.89, 46.21, 40.11, 42.88, 50.63])
PrS_slms = np.array([17.38, 28.47, 38.82, 47.32, 42.51, 45.84, 52.79])
InS_llms = np.array([32.73, 42.67, 48.56, 56.83, 54.43, 57.31, 68.43])
InS_slms = np.array([29.38, 41.73, 49.76, 58.39, 55.16, 58.75, 72.56])
PrL_llms = np.array([22.00, 31.98, 42.70, 50.64, 48.24, 51.20, 57.12])
PrL_slms = np.array([19.78, 31.98, 42.51, 51.39, 51.02, 54.71, 61.25])
InL_llms = np.array([35.85, 46.04, 53.60, 60.79, 61.99, 64.15, 70.98])
InL_slms = np.array([32.01, 44.96, 53.96, 62.35, 62.47, 66.31, 73.27])
gsm8k_llms = np.array([40.26, 62.32, 76.12, 76.12, 87.79, 87.79, 91.05])
gsm8k_slms = np.array([40.71, 65.35, 76.57, 82.03, 88.17, 89.61, 91.36])
math_llms = np.array([16.32, 24.06, 26.44, 38.14, 49.94, 55.02, 58.83])
math_slms = np.array([16.26, 27.84, 30.92, 43.78, 52.22, 55.28, 60.75])
humaneval_llms = np.array([30.49, 50.00, 63.41, 70.73, 75.00, 80.49, 82.93])
humaneval_slms = np.array([34.76, 52.44, 64.02, 71.95, 75.61, 81.71, 84.67])
mbpp_llms = np.array([27.60, 43.20, 55.40, 61.60, 67.20, 71.20, 76.00])
mbpp_slms = np.array([28.00, 49.94, 55.80, 61.80, 67.20, 73.20, 76.80])

# 绘制每个子图
plot_single(axes[0][0], x, PrS_llms, PrS_slms, x, 'Pr.(S)', color[1], color[0])
plot_single(axes[0][1], x, InS_llms, InS_slms, x, 'In.(S)', color[1], color[0])
plot_single(axes[0][2], x, PrL_llms, PrL_slms, x, 'Pr.(L)', color[1], color[0])
plot_single(axes[0][3], x, InL_llms, InL_slms, x, 'In.(L)', color[1], color[0])
plot_single(axes[1][0], x, gsm8k_llms, gsm8k_slms, x, 'GSM8K', color[1], color[0])
plot_single(axes[1][1], x, math_llms, math_slms, x, 'MATH', color[1], color[0])
plot_single(axes[1][2], x, humaneval_llms, humaneval_slms, x, 'HumanEval', color[1], color[0])
plot_single(axes[1][3], x, mbpp_llms, mbpp_slms, x, 'MBPP', color[1], color[0])

# 调整布局以避免子图之间的重叠
plt.tight_layout()

# 保存图表为PDF文件
plt.savefig('figs/scaling.png')
plt.savefig('figs/scaling.pdf')