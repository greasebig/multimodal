# 第一次默写
# 1.multi head attn
class multi_head_attn(nn.modeule):
    def __init__(self, d_model, head)#def __init__(q, k, v, head):
        super(multi_head_attn, self).__init__()#super(multi_head_attn)
        #self.q = q
        #self.k = k
        #self.v = v
        self.w_q = nn.Linear(d_model, d_model)#self.w_q = 
        self.w_k = nn.Linear(d_model, d_model)#self.w_k = 
        self.w_v = nn.Linear(d_model, d_model)#self.w_v = 
        self.w_concat = nn.Linear(d_model, d_model)#self.w_concat = 
        self.scaleDotSelfAttn = scaleDotSelfAttn()
        self.head = head#
        
        
    def forward(self, q, k, v, mask=None):
        q, k ,v = self.w_q(q), self.w_k(k), self.w_v(v)#q, k ,v = w_q(q), w_k(k), w_v(v)
        q, k ,v = self.split(q), self.split(k), self.split(v)
        out = scaleDotSelfAttn(q, k, v, mask)#out = scaleDotSelfAttn(q, k, v)
        out = self.w_concat(self.concat(out))#out = w_concat(out)
        return out
        
    def concat(self, k):#def concat(k):
        batch, length, head, d_tensor = k.size()
        k = k.transpose(1, 2).contiguous.view(batch, length, self.head * d_tensor)
        #k.transpose(2, 3).congious.view(batch, length, head * d_tensor)
        return k 
        
    def split(self, k):#def split(k):
        batch, length, d_model = k.size()
        k = k.view(batch, length, self.head, d_model//self.head).transpose(1, 2)
        #k.view(batch, length, head, d_model//head).transpose(2, 3)
        return k 


class scaleDotSelfAttn(nn.modeule):
    def __init__(self):#def __init__(q,k,v):
        super(scaleDotSelfAttn, self).__init__()#super(scaleDotSelfAttn)
        #self.q = q
        #self.k = k
        #self.v = v
        self.softmax = nn.softmax(dim=-1) #self.softmax = 
        
        
    def forward(self, q, k ,v ,mask=None):#def forward():
        batch, length, head, d_tensor = k.size()
        k_t = k.transpose(2,3)
        score = q @ k_t / math.sqrt(d_tensor)#score = q @ k_t / sqrt(d_tensor)
        if mask is not None :
            score = score.masked_fill(mask == 0, -10000)#score = score.mask_fills(mask,)
        #目的是将score张量中在mask张量中对应位置为0的元素值替换为-10000。
        #这种操作通常用于在计算softmax等操作时，将不需要的部分设置为一个较大的负数，
        #以确保在计算指数时这些位置的值趋近于零。这有助于避免数值稳定性的问题。
        scored = self.softmax(score)#scored = softmax(score)
        out = scored @ v
        return out





# 2.搭建简单神经网络

# 原始
dataset1 = 
data = train_loader(dataset1)

model = 

optimizer = 

model.train()
for i, (train_data, label) in enumerate(data):
    no_grad
    pred = model(train_data)
    loss = nn.CrossEntropyLoss(pred, label)
    loss.backward()
    optimizer.step()
    print(loss.item())


# 修改

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = F.sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

input_size = 10
output_size = 1
X = torch.randn(1000,input_size)
y = torch.randint(0, 3, (1000, output_size), dtype = torch.float32)

dataset1 = 
data = DataLoader(dataset1, batch_size = 64, shuffle = True)

model = SimpleNN(input_size, hidden_size=64, output_size)

optimizer = optim.SGD(model.parameters(), lr=0.01)

model.train()
for epoch in range(10):
    for i, (train_data, label) in enumerate(data):
        optimizer.zero_grad()
        pred = model(train_data)
        loss = nn.CrossEntropyLoss(pred, label)
        loss.backward()
        optimizer.step()
    print(loss.item())






# 3.写出交叉熵计算，使用torch和不使用torch
import numpy as np
def MyCSLoss(predict, target):
    predict = np.exp(predict - np.max(predict, axis=1, keepdims=True))
    predict /= np.sum(predict, axis=1, keepdims=True)

    column_index = np.arange(predict.shape[0])
    predict = predict[column_index, target]

    return -np.log(predict)

predict = np.array([[4, 5, 2], [1, 3, 4], [0, 1, 2]])
target = np.array([0, 1, 2])
loss = MyCSLoss(predict2, target2)
print(loss)


# 4.自己定义数据加载
import torch
from torch.utils.data import DataLoader, Dataset
class CustomDataset(Dataset):
    def __init__(self, data_root, label_root):
        super(CustomDataset, self).__init__()
        self.data = data_root
        self.label = label_root

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__():
        return len(data)



