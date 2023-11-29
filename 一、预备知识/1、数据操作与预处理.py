import torch
import os
import pandas as pd

print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&数据操作&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
# # region
# print("##########################张量创建与访问###############################")
# x = torch.arange(12)  # 生成一个张量
# print(x)
# print(x.shape)  # 可以通过shape属性访问张量的形状
# print(x.numel())  # 可以通过numel()函数访问张量元素的总数
# x = x.reshape(3, 4)  # 改变一个张量的形状而不改变元素数量和元素值，可以调用reshape函数
# print(x)
# y = torch.zeros((2, 3, 4))  # 创建3维的二通道三行四列的全零张量
# print(y)
# y = torch.ones((2, 3, 4))  # 创建3维的二通道三行四列的全一张量
# print(y)
# # 通过提供包含数值的Python列表(或嵌套列表)来为所需张量中的每个元素赋予确定值
# z = torch.tensor([[2, 1, 3, 4], [2, 3, 4, 1], [3, 1, 2, 4]])
# print(z)
# print("#############################张量运算################################")
# # 常见的标准算术运算符+、-、*、/和**都可以被升级为按元素运算
# x = torch.tensor([1.0, 2, 3])
# y = torch.tensor([2, 2, 2])
# print(x + y)
# print(x - y)
# print(x * y)
# print(x / y)
# print(x ** y)
# # 张量连结（concatenate）
# X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(torch.cat((X, Y), dim=0))  # 在第一维度进行合并，即在行的维度进行合并
# print(torch.cat((X, Y), dim=1))  # 在第二维度进行合并，即在列的维度进行合并
# # 按元素判断x是否与y相同
# print(X == Y)
# # 对张量中的所有元素进行求和，会产生一个单元素
# A = X.sum()
# print(A)
# print("##########################广播机制###############################")
# a = torch.arange(3).reshape((3, 1))
# b = torch.arange(2).reshape((1, 2))
# print(a, '\n', b)
# print(a + b)  # 由于a和b分别是3*1和1*2矩阵，将两个矩阵广播为一个更大的矩阵.矩阵a将复制列，矩阵b将复制行，然后再按元素相加。
# print("##########################索引和切片###############################")
# print(X)
# print(X[-1])  # 访问最后一行
# print([X[1:3]])  # 左闭右开，输出1—2行，不包括3
# X[1, 2] = 9  # 修改X张量的位于1，2位置的数字为9
# print(X)
# X[0:2, :] = 12  # 修改0—1行的所有列为12
# print(X)
# print("##########################节省内存###############################")
# before = id(Y)  # id类似于指针，在python中为标识号
# Y = Y + X
# print(id(Y) == before)  # id不同会重新分配内存
# # 原地操作
# Z = torch.zeros_like(Y)
# print('id(Z):', id(Z))  # 输出z的id
# Z[:] = X + Y
# print('id(Z):', id(Z))  # 重新输出z的id，对比之前id无变化
# print("######################转换为其他Python对象########################")
# A = X.numpy()  # 将torch张量转换成numpy张量
# B = torch.tensor(A)  # 将numpy张量转换成torch张量
# print(type(A), type(B))
# a = torch.tensor([3.5])
# print(a, a.item(), float(a), int(a))  # 其他类型转化。item()函数的作用是从包含单个元素的张量中取出该元素值，并保持该元素的类型不变。
# # endregion
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&数据预处理&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
# region
print("##########################读取数据集###############################")
# 在根目录下创建一个预备知识，exist_ok=True表示如果目录已经存在，不会引发FileExistsError。
os.makedirs(os.path.join("../预备知识"), exist_ok=True)
# 在目录../1、数据操作 + 数据预处理下创建一个house_tiny.csv文件
data_file = os.path.join("../预备知识", 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
# 使用pandas读取数据库
data = pd.read_csv(data_file)
print(data)
print("##########################处理缺失值###############################")
# ##########################方法1：插值################################
# iloc是integer location简写，他是pandas数组专用的切片函数
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs.iloc[:, 0:1] = inputs.iloc[:, 0:1].fillna(inputs.iloc[:, 0:1].mean())
print(inputs)
# #####################方法2：将NaN视为一个类别##########################
# pd.get_dummies()是pandas库中的一个函数，用于对分类变量进行独热编码。独热编码是一种将分类数据变量转换为二进制向量的过程。
# 对于inputs中的类别值或离散值，我们将“NaN”视为一个类别。由于“Alley”列只接受两种类型的类别值“Pave”和“NaN”， pandas
# 可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。“Alley”类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”
# 的值设置为0。缺少“Alley”类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
print("##########################转换为张量格式###############################")
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X)
print(y)
# endregion
