import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import gaussian_kde

color = ['#1f77b4', '#ff7f0e']

# 加载txt文件中的logprob数据
def load_logprob_data(file_path):
    logprobs = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    for d in data:
        logprobs.append(float(list(d.values())[0]))
    return logprobs

# 将对数概率转换为概率
def logprob_to_prob(logprobs):
    return np.exp(logprobs)

# 加载数据
file_path1 = 'results/logprobs-large.json'  # 第一个文件路径
file_path2 = 'results/logprobs-small.json'  # 第二个文件路径
logprobs1 = load_logprob_data(file_path1)
logprobs2 = load_logprob_data(file_path2)

# 将对数概率转换为概率
probs1 = logprob_to_prob(logprobs1)
probs2 = logprob_to_prob(logprobs2)

# 创建图形和坐标轴用于绘制
fig, ax = plt.subplots(figsize=(4, 3))
ax.grid(linestyle="-.")

# 绘制平滑曲线
kde1 = gaussian_kde(probs1)
kde2 = gaussian_kde(probs2)

x = np.linspace(0, 1, 1000)  # 从0到1的概率范围
ax.plot(x, kde1(x), label='LLMs', color=color[1])
ax.plot(x, kde2(x), label='SLMs', color=color[0])

# 反转x轴的方向
ax.invert_xaxis()

ax.set_xlabel('Probability', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('figs/logprob.pdf')
# plt.show()  # 可选，显示图形