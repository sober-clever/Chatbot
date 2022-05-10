# 本程序将根据一系列数据对，拟合出一个线性函数
# 数据对大致符合 y = 5x + 7
# 需要做的是传入训练数据，选择训练模型为线性模型，选择损失函数，选择优化器
import torch
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD
import numpy as np

x = np.random.rand(256) # 输入值，随机生成的 256 个数字
noise = np.random.randn(256) / 4 # 给结果加一些扰动，模拟实际获得的数据对
y = x * 5 + 7 + noise # 实际输出值，将作为预期输出


model=Linear(1, 1)  # 预期模型，线性模型
criterion = MSELoss() # 损失函数：均方损失函数
optim = SGD(model.parameters(), lr = 0.01) # 优化器，选用 SGD
# 优化器的作用就是每次更新参数，参数即为 model.parameters
epochs = 3000 # 训练 3000 次
x_train = x.reshape(-1, 1).astype('float32') # astype('float32') 是为了下一步可以直接转换为 torch.float.
y_train = y.reshape(-1, 1).astype('float32')

for i in range(epochs):
    # 整理输入和输出的数据，这里输入和输出一定要是torch的Tensor类型
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)

    outputs = model(inputs) # 使用模型进行预测

    optim.zero_grad() # 梯度置0，否则会累加

    loss = criterion(outputs, labels) # 计算损失

    loss.backward() # 反向传播

    optim.step() # 使用优化器默认方法优化

    if (i%100==0):
        #每 100次打印一下损失函数，看看效果
        print('epoch {}, loss {:1.4f}'.format(i,loss.data.item()))

[w, b] = model.parameters() # 实际训练出来的参数，也即我们得到的公式为 y = w*x + b
print (w.item(),b.item())