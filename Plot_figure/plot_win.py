import matplotlib.pyplot as plt
import numpy as np

# 配置数据
configs = ['5%', '10%', '15%']
win_nums = [421, 445, 434]  # 每个配置的赢的次数
tie_nums = [57, 50, 76]     # 每个配置的平局次数
lose_nums = [327, 310, 295]  # 每个配置的输的次数

# 颜色设置
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 设置柱子的宽度
bar_width = 0.50

# 设置x轴的位置
x_positions = np.arange(len(configs))

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))

# 绘制第一个子图
bottom1 = ax1.barh(x_positions, win_nums, bar_width, label='Win', color=color[0])
bottom2 = ax1.barh(x_positions, tie_nums, bar_width, label='Tie', color=color[1], left=win_nums)
bottom3 = ax1.barh(x_positions, lose_nums, bar_width, label='Lose', color=color[2], left=np.array(win_nums) + np.array(tie_nums))

# 添加文本标签
for i in range(len(x_positions)):
    # 让每个数字标签显示在它们对应的柱子中间
    ax1.text(win_nums[i] / 2, x_positions[i], str(win_nums[i]), va='center', ha='center', color='black', fontsize=12)
    ax1.text(win_nums[i] + tie_nums[i] / 2, x_positions[i], str(tie_nums[i]), va='center', ha='center', color='black', fontsize=12)
    ax1.text(win_nums[i] + tie_nums[i] + lose_nums[i] / 2, x_positions[i], str(lose_nums[i]), va='center', ha='center', color='black', fontsize=12)

ax1.set_yticks(x_positions)
ax1.set_yticklabels(configs, fontsize=14)
ax1.set_title("IC-IFD vs. IFD on Llama-3-8B", fontsize=14)
ax1.set_xticklabels(['0', '100', '200', '300', '400', '500', '600', '700', '800'], fontsize=14)

# 在第一个子图中添加legend，并调整位置
ax1.legend(loc='upper left', ncol=1, fontsize=12)

configs = ['5%', '10%', '15%']
win_nums = [412, 428, 435]  # 每个配置的赢的次数
tie_nums = [55, 58, 58]     # 每个配置的平局次数
lose_nums = [338, 319, 312]  # 每个配置的输的次数

# 绘制第二个子图
bottom1 = ax2.barh(x_positions, win_nums, bar_width, label='Win', color=color[0])
bottom2 = ax2.barh(x_positions, tie_nums, bar_width, label='Tie', color=color[1], left=win_nums)
bottom3 = ax2.barh(x_positions, lose_nums, bar_width, label='Lose', color=color[2], left=np.array(win_nums) + np.array(tie_nums))

# 添加文本标签
for i in range(len(x_positions)):
    # 让每个数字标签显示在它们对应的柱子中间
    ax2.text(win_nums[i] / 2, x_positions[i], str(win_nums[i]), va='center', ha='center', color='black', fontsize=12)
    ax2.text(win_nums[i] + tie_nums[i] / 2, x_positions[i], str(tie_nums[i]), va='center', ha='center', color='black', fontsize=12)
    ax2.text(win_nums[i] + tie_nums[i] + lose_nums[i] / 2, x_positions[i], str(lose_nums[i]), va='center', ha='center', color='black', fontsize=12)

ax2.set_yticks(x_positions)
ax2.set_yticklabels(configs, fontsize=14)
ax2.set_title("IC-IFD vs. IFD on Llama-3.2-3B", fontsize=14)
ax2.set_xticklabels(['0', '100', '200', '300', '400', '500', '600', '700', '800'], fontsize=14)

# 在第二个子图中添加legend，并调整位置
ax2.legend(loc='upper left', ncol=1, fontsize=12)

# 调整子图间距
plt.tight_layout()

# 显示图形
plt.savefig("figs/win.pdf")
plt.savefig("figs/win.png")
