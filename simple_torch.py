'''
定义神经网络模型：
使用torch.nn.Module创建一个继承类，定义神经网络的结构。
在类的构造函数__init__中定义神经网络的层，如全连接层 (nn.Linear)。
实现forward方法，定义数据在网络中的流动方式。

定义数据：
生成训练数据，可以使用随机数据或从文件加载。
将数据封装成PyTorch的Dataset和DataLoader，以便进行批量训练。

初始化模型、损失函数和优化器：
实例化定义好的神经网络模型。
选择合适的损失函数，这里使用了二分类交叉熵损失 (nn.BCELoss)。
选择优化器，这里使用了随机梯度下降 (optim.SGD)。

训练模型：
设置训练的轮数 (num_epochs)。
使用嵌套的循环，外循环是迭代的轮数，内循环是每个epoch中的每个小批次。
在每个小批次中进行：
梯度清零 (optimizer.zero_grad())。
前向传播计算输出。
计算损失。
反向传播计算梯度。
更新权重 (optimizer.step())

使用训练好的模型进行预测：
对测试数据进行前向传播，得到模型的输出。
在实际应用中，可以使用这些输出进行相应的预测或评估任务。

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Step 1: 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)

# Step 2: 定义数据
# 生成一些随机的训练数据
num_samples = 1000
input_size = 10
output_size = 1

# 创建随机输入和标签
X = torch.randn(num_samples, input_size)
y = torch.randint(0, 2, (num_samples, output_size), dtype=torch.float32)

# 将数据封装成PyTorch的Dataset和DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Step 3: 初始化模型、损失函数和优化器
model = SimpleNN(input_size, hidden_size=64, output_size=output_size)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 4: 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # 将梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 5: 使用训练好的模型进行预测
# 假设有一些测试数据X_test，你可以用训练好的模型进行预测
# test_outputs = model(X_test)
