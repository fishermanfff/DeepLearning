import matplotlib.pyplot as plt
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6  # 定义一个六面骰子的公平概率分布
print(multinomial.Multinomial(1, fair_probs).sample())  # 从具有1次试验的多项式分布中使用公平概率生成一个样本
print(multinomial.Multinomial(10, fair_probs).sample())  # 从具有10次试验的多项式分布中使用公平概率生成一个样本
# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts / 1000)  # 相对频率作为估计值
# 这一行生成了500组试验，每组试验抛掷一个六面骰子10次，记录每个面出现的次数。
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)  # 在第零维度求和
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)  # 各列除以总数

d2l.set_figsize((6, 4.5))  # 设置图像比例
# 循环遍历六个骰子面，对每个面的概率估算进行绘制。黑色虚线表示每个面的真实概率（1/6）
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
# 绘图设置
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
plt.show()
