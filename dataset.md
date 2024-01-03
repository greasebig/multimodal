# Pytorch加载数据集的方式总结

## 自己重写定义（Dataset、DataLoader）
我们有自己制作的数据以及数据标签，但是有时候感觉不太适合直接用Pytorch自带加载数据集的方法。我们可以自己来重写定义一个类，这个类继承于 torch.utils.data.Dataset，同时我们需要重写这个类里面的两个方法 _ getitem__ () 和__ len()__函数。   

```python
import torch
import numpy as np

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

# 随机生成数据，大小为10 * 20列
source_data = np.random.rand(10, 20)
# 随机生成标签，大小为10 * 1列
source_label = np.random.randint(0,2,(10, 1))
# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
torch_data = GetLoader(source_data, source_label)

```   
构造了一个数据加载器torch_data，但是还是不能直接传入网络中。接下来需要构造数据装载器，产生可迭代的数据，再传入网络中   
```
torch.utils.data.DataLoader(dataset,batch_size,shuffle,drop_last，num_workers)

1.dataset     : 加载torch.utils.data.Dataset对象数据
2.batch_size  : 每个batch的大小,将我们的数据分批输入到网络中
3.shuffle     : 是否对数据进行打乱
4.drop_last   : 是否对无法整除的最后一个datasize进行丢弃

```  
```python
...
torch_data = GetLoader(source_data, source_label)

from torch.utils.data import DataLoader
datas = DataLoader(torch_data, batch_size = 4, shuffle = True, drop_last = False, num_workers = 2)
for i, (data, label) in enumerate(datas):
	# i表示第几个batch， data表示batch_size个原始的数据，label代表batch_size个数据的标签
    print("第 {} 个Batch \n{}".format(i, data))


```   

## 用Pytorch自带的类（ImageFolder、datasets、DataLoader）
### ImageFolder
```
A generic data loader where the images are arranged in this way:

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png


dataset=torchvision.datasets.ImageFolder(
                       root, transform=None, 
                       target_transform=None, 
                       loader=<function default_loader>, 
                       is_valid_file=None)


1.root：根目录，在root目录下，应该有不同类别的子文件夹；
    |--data(root)
        |--train
            |--cat
            |--dog
        |--valid
            |--cat
            |--dog        
2.transform：对图片进行预处理的操作，原始图像作为一个输入，返回的是transform变换后的图片；
3.target_transform：对图片类别进行预处理的操作，输入为 target，输出对其的转换。 如果不传该参数，即对target不做任何转换，返回的顺序索引 0,1, 2…
4.loader：表示数据集加载方式，通常默认加载方式即可；
5.is_valid_file：获取图像文件的路径并检查该文件是否为有效文件的函数(用于检查损坏文件)


（1）self.classes：用一个 list 保存类别名称

（2）self.class_to_idx：类别对应的索引，与不做任何转换返回的 target 对应

（3）self.imgs：保存(img_path, class) tuple的list

```  

#### ImageFolder加载数据集完整例子
```python
# 1.导入相关数据库
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets

# 2.定义图片转换方式
train_transforms = torchvision.transforms.Compose([
	transforms.RandomResizedCrop(400),
	transforms.ToTensor()
])

# 3. 定义地址
path = os.path.join(os.getcwd(), 'data', 'train')

# 4. 将文件夹数据导入
dataset = ImageFolder(root=path, transform=train_transforms)

# 5. 将文件夹数据导入
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size = batch_size, shuffle=True,
                                           num_workers = 2)
# 6. 传入网络进行训练
for epoch in range(epochs):
    train_bar = tqdm(train_loader, file = sys.stdout)
    for step, data in enumerate(train_bar):
    ...

```      

### 加载常见的数据集
有些数据集是公共的，比如常见的MNIST，CIFAR10，SVHN等等。这些数据集在Pytorch中可以通过代码就可以下载、加载。如下代码所示。用torchvision中的datasets类下载数据集，并还是结合DataLoader来构建可直接传入网络的数据装载器。   
```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([
        					transforms.Resize((input_size, input_size)), 
       					    transforms.ToTensor(), 
        					transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)


```


## transforms变换
torchvision.transforms是Pytorch中的图像预处理包。一般定义在加载数据集之前，用transforms中的Compose类把多个步骤整合到一起，而这些步骤是transforms中的函数。   

函数	含义   
transforms.Resize	把给定的图片resize到given size
transforms.Normalize	用均值和标准差归一化张量图像
transforms.Totensor	可以将PIL和numpy格式的数据从[0,255]范围转换到[0,1] ; <br /另外原始数据的shape是（H x W x C），通过transforms.ToTensor()后shape会变为（C x H x W）
transforms.RandomGrayscale	将图像以一定的概率转换为灰度图像
transforms.ColorJitter	随机改变图像的亮度对比度和饱和度
transforms.Centercrop	在图片的中间区域进行裁剪
transforms.RandomCrop	在一个随机的位置进行裁剪
transforms.FiceCrop	把图像裁剪为四个角和一个中心
transforms.RandomResizedCrop	将PIL图像裁剪成任意大小和纵横比
transforms.ToPILImage	convert a tensor to PIL image
transforms.RandomHorizontalFlip	以0.5的概率水平翻转给定的PIL图像
transforms.RandomVerticalFlip	以0.5的概率竖直翻转给定的PIL图像
transforms.Grayscale	将图像转换为灰度图像

### torch.linspace(1, 10, 10)
start end step样本数  
线性空间

## 简单例子
```python
import torch
from torch.utils.data import random_split
import torch.utils.data as Data
train_x = torch.randn(10,8)
train_y = torch.randn(10,2)
#正态分布取样，行列
dataset = Data.TensorDataset(train_x,train_y)  #把训练集和标签继续封装

train_data,eval_data=random_split(dataset,[round(0.8*train_x.shape[0]),round(0.2*train_x.shape[0])],generator=torch.Generator().manual_seed(42))  #把数据机随机切分训练集和验证集
for i in train_data:
    print(i)
loader = Data.DataLoader(dataset = train_data, batch_size = 2, shuffle = True, num_workers = 0 , drop_last=False)
for step,(train_x,train_y) in enumerate(loader):
    print(step,':',(train_x,train_y))

```
