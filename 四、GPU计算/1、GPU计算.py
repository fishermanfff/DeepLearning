import torch
from torch import nn


print(torch.cuda.device_count())
torch.device('cuda')  # 启用GPU计算


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


print(try_gpu(), try_gpu(1), try_all_gpus())

# ####################################################张量与GPU##########################################
x = torch.tensor([1, 2, 3])
print("张量存在于：", x.device)
x = torch.tensor([1, 2, 3], device=try_gpu())
print("张量存在于：", x.device)
X = torch.ones(2, 3, device=try_gpu())
# ####################################################神经网络与GPU######################################
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(X))
print(net[0].weight.data.device)
