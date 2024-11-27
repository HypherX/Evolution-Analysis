import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.ticker import FuncFormatter

color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 加载txt文件中的logprob数据
def load_logprob_data(file_path):
    logprobs = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    for d in data:
        logprobs.append(float(list(d.values())[0]))
    return logprobs

# 自定义格式化函数，将y轴数值转换为百分比
def format_yaxis(value, tick_position, total_count):
    percentage = (value / total_count) * 100
    return f'{percentage:.0f}%'

# 绘制直方图的函数
def plot_histogram(ax, logprobs, label, color):
    min_logprob = min(logprobs)
    max_logprob = max(logprobs)
    bins = np.linspace(min_logprob, max_logprob, 25)
    hist, bin_edges = np.histogram(logprobs, bins=bins)
    total_count = sum(hist)
    ax.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]), color=color, label=label)
    ax.set_xlabel('Log Probability', fontsize=14)
    ax.set_ylabel('Proportion', fontsize=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: format_yaxis(v, p, total_count)))

# 加载数据
file_path1 = 'results/logprobs-large.json'  # 第一个文件路径
file_path2 = 'results/logprobs-small.json'  # 第二个文件路径
logprobs1 = load_logprob_data(file_path1)
logprobs2 = load_logprob_data(file_path2)

# 创建图形和坐标轴用于绘制
fig, ax = plt.subplots(figsize=(4, 3))
ax.grid(linestyle="-.")

# 绘制数据
plot_histogram(ax, logprobs2, 'SLMs', color[0])
plot_histogram(ax, logprobs1, 'LLMs', color[1])


ax.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('figs/logprob.pdf')
# plt.show()  # 可选，显示图形