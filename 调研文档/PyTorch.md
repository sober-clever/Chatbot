## 环境搭建

#### 安装 Pytorch

```shell
pip install numpy==1.16.4
pip3 install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
```

验证输入python 进入

```python
import torch
torch.__version__
# 得到结果'1.5.0'
```

#### 配置 Jupyter Notebook

新建的环境是没有安装 ipykernel 的, 所以无法注册到Jupyter Notebook中，先要准备下环境

```bash
#安装ipykernel
conda install ipykernel
#写入环境
python -m ipykernel install  --name pytorch --display-name "Pytorch for Deeplearning"
```

下一步就是定制 Jupyter Notebook

```bash
#切换回基础环境
activate base
#创建jupyter notebook配置文件
jupyter notebook --generate-config
## 这里会显示创建jupyter_notebook_config.py的具体位置
```

打开文件，修改

```
c.NotebookApp.notebook_dir = '' 默认目录位置
c.NotebookApp.iopub_data_rate_limit = 100000000 这个改大一些否则有可能报错
```





## 张量

张量的英文是 Tensor，它是 PyTorch 里面基础的运算单位，与 Numpy 的 ndarray 相同，都表示一个**多维的矩阵**。

PyTorch 的 Tensor 可以在 GPU 上运行，这大大加快了运算速度。

> 张量（Tensor）是一个定义在一些向量空间和一些对偶空间的笛卡儿积上的多重线性映射，其坐标是|n|维空间内，有|n|个分量的一种量， 其中每个分量都是坐标的函数， 而在坐标变换时，这些分量也依照某些规则作线性变换。r 称为该张量的秩或阶（与矩阵的秩和阶均无关系）。 (来自百度百科)

生成一个简单的张量并查看其大小

```python
x = torch.rand(2, 3)
print(x.shape)
print(x.size()) # 也可以使用size()函数，返回的结果都是相同的
```

在同构的意义下，第零阶张量 （r = 0） 为标量 （Scalar），第一阶张量 （r = 1） 为向量 （Vector）， 第二阶张量 （r = 2） 则成为矩阵 （Matrix），第三阶以上的统称为多维张量。

对于标量，我们可以直接使用 .item() 从中取出其对应的python对象的数值

##### 常用方法

```python
# 沿着行取最大值
max_value, max_idx = torch.max(x, dim=1)

# 每行 x 求和
sum_x = torch.sum(x, dim=1)

# 两个张量直接求和
z = x + y
```



## 使用 PyTorch 计算梯度数值

PyTorch 的 **Autograd** 模块实现了深度学习的算法中的向传播求导数，在张量（Tensor类）上的所有操作，**Autograd** 都能为他们自动提供微分，简化了手动计算导数的复杂过程。

### Autograd

在张量创建时，通过设置 requires_grad 标识为 Ture 来告诉 Pytorch 需要对该张量进行自动求导，PyTorch 会记录该张量的每一步操作历史并自动计算

```python
x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
z = torch.sum(x+y)
print(z)
z.backward()
print(x.grad)
```

每个张量都有一个 .grad_fn 属性，如果这个张量是用户手动创建的那么这个张量的 grad_fn 是 None。



## 神经网络包 nn 和优化器 optm

torch.nn 是专门为神经网络设计的模块化接口，nn 构建于 Autograd 之上，可用来定义和运行神经网络。

nn.functional，这个包中包含了神经网络中使用的一些常用函数，这些函数的特点是，不具有可学习的参数(如 ReLU，pool，DropOut 等)，这些函数可以放在构造函数中，也可以不放，但是这里建议不放。

注：一般情况下我们会将 nn.functional 设置为大写的 F，这样缩写方便调用





## 参考文献

[PyTorch 中文手册（pytorch handbook） - Pytorch中文手册](https://handbook.pytorch.wiki/index.html)

