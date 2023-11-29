import torch
print("###############################################线性代数基础#######################################################")
# region
print("#####################################标量######################################")
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y, x * y, x / y, x**y)
print("#####################################向量######################################")
x = torch.arange(4)
print(x, x[3])
print(len(x), x.shape)
print("#####################################矩阵######################################")
A = torch.arange(20).reshape(5, 4)
print(A, A.T)  # 输出矩阵及其转置矩阵
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)
print("#####################################张量######################################")
X = torch.arange(24).reshape(2, 3, 4)
print(X)
print("################################张量算法的基本性质################################")
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print(A, A + B)
print("A*B=", A * B)
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X, (a * X).shape)
print("#####################################降维######################################")
x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())
print(A)
print(A.shape, A.sum())
A_sum_axis0 = A.sum(axis=0)  # 按第0维度求和（列）
print(A_sum_axis0, A_sum_axis0.shape)
A_sum_axis1 = A.sum(axis=1)  # 按第1维度求和（行）
print(A_sum_axis1, A_sum_axis1.shape)
print(A.sum(axis=[0, 1]))  # 按两个维度求和，结果和A.sum()相同
print(A.mean(), A.sum() / A.numel())
print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])
print("###################################非降维求和####################################")
sum_A = A.sum(axis=1, keepdims=True)  # 保持维度求和
print(sum_A)
print(A / sum_A)
print(A.cumsum(axis=0))  # 在第0维度累加求和
print("###################################点积#########################################")
y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))
print(torch.sum(x * y))
print("###################################矩阵-向量积###################################")
print(A, x)
print(A.shape, x.shape, torch.mv(A, x))
print("###################################矩阵-矩阵乘法###################################")
B = torch.ones(4, 3)
print(A, B)
print(torch.mm(A, B))
print("######################################范数######################################")
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))  # 向量L1范数，根号下所有元素平方和
print(torch.abs(u).sum())  # 向量L2范数，所有元素绝对值之和
print(torch.norm(torch.ones((4, 9))))  # 矩阵范数
print("######################################范数和目标#################################")
# endregion
print("#################################################自动求导#########################################################")
# region
print("######################################例子##########################################")
x = torch.arange(4.0)
print(x)  # 输出生成的向量
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)，即启用梯度计算
print(x.grad)  # 默认值是None
y = 2 * torch.dot(x, x)  # dot操作为对位相乘，即内积，y=2*xT*x
print(y)
y.backward()  # 对y的x变量反向传播求导
print(x.grad)  # 输出导数
print(x.grad == 4 * x)  # 判断倒数是否正确
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()  # 清除梯度累计
y = x.sum()  # 即y=x1+x2+x3+x4
y.backward()
print(x.grad)
print("######################################非标量变量的反向传播##########################################")
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
print(y)
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)
print("######################################分离计算##########################################")
x.grad.zero_()
y = x * x
u = y.detach()  # 把y detach，即把y当成常数
z = u * x
z.sum().backward()
print(x.grad == u)
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)
print("######################################Python控制流的梯度计算######################################")
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)
# endregion
