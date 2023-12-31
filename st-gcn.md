本篇讲解的是 STGCN，不是 ST-GCN！前者是用于「交通流量预测」，后者是用于「人体骨骼的动作识别」。名字很像，但是模型不一样。
## st-gcn论文
AAAI2018||ST-GCN：Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition    
图卷积网络（Graph Convolutional Network，GCN）借助图谱的理论来实现空间拓扑图上的卷积，提取出图的空间特征，具体来说，就是将人体骨骼点及其连接看作图，再使用图的邻接矩阵、度矩阵和拉普拉斯矩阵的特征值和特征向量来研究该图的性质。   
ST-GCN单元通过GCN学习空间中相邻关节的局部特征，而时序卷积网络（Temporal convolutional network，TCN）则用于学习时间中关节变化的局部特征。卷积核先完成一个节点在其所有帧上的卷积，再移动到下一个节点，如此便得到了骨骼点图在叠加下的时序特征。    
![Alt text](assets_picture/stgcn/image.png)  
在ST-GCN的网络中，采取每一帧的点的坐标信息和相邻帧的相同点的坐标变换信息作为双流输入，分别采用两个不共享的GCN提取各自特征信息，并进行特征融合后采用softmax函数计算。  

核心观点是将TCN与GCN相结合,用来处理有时序关系的图结构数据。网络分为2个部分:GCN_Net与TCN_Net。  
GCN_Net对输入数据进行空间卷积,即不考虑时间的因素,卷积作用于同一时序的不同点的数据。TCN_Net对数据进行时序卷积,考虑不同时序同一特征点的关系,卷积作用于不同时序同一点的数据。  
## 实现
对于序列数据的维度要求为(N,C,T,V,M)。维度要求以及详细说明如下表：  
维度	大小	说明  
N	不定	数据集序列个数  
C	2	关键点坐标维度，即(x, y)  
T	50	动作序列的时序维度（即持续帧数）  
V	17	每个人物关键点的个数  
M	1	人物个数，这里我们每个动作序列只针对单人预测    
ST-GCN输入的格式为(1,3,300,18,2)，对应于(batch,channel,frame,joint,person)。  
输出将是（batch，class，output_frame，joint，person）的置信值  


方案说明
1. 使用多目标跟踪获取视频输入中的行人检测框及跟踪ID序号，模型方案为PP-YOLOE，详细文档参考PP-YOLOE，跟踪方案为BOT-SORT
2.	通过行人检测框的坐标在输入视频的对应帧中截取每个行人。
3.	使用关键点识别模型得到对应的17个骨骼特征点。骨骼特征点的顺序及类型与COCO一致.
4.	每个跟踪ID对应的目标行人各自累计骨骼特征点结果，组成该人物的时序关键点序列。当累计到预定帧数或跟踪丢失后，使用行为识别模型判断时序关键点序列的动作类型。当前版本模型支持摔倒行为的识别，预测得到的class id对应关系为：
0: 摔倒
1: 其他


## 动手实践
### 实践问题
paddlevideo训练st-gcn   
1.
```
File "/data/lujunda/drown/code/PaddleVideo/paddlevideo/tasks/train.py", line 286, in train_model
    batch_size * record_list["batch_time"].count /
ZeroDivisionError: division by zero
```
batch过大  
2.  
使用paddle 2.3 或 2.4   
2.6和2.1不行   
3.  
使用ppyoloe+准备数据时，会发生键名错误   
```
python deploy/python/det_keypoint_unite_infer.py \
--det_model_dir=output_inference/ppyoloe_plus_crn_x_80e_coco \
--keypoint_model_dir=output_inference/dark_hrnet_w32_256x192 \
--video_file=../work/clip/0619_12.mp4 \
--device=GPU --save_res=True

Traceback (most recent call last):
  File "/data/lujunda/drown/code/PaddleDetection-2.5.0/deploy/python/det_keypoint_unite_infer.py", line 376, in <module>
    main()
  File "/data/lujunda/drown/code/PaddleDetection-2.5.0/deploy/python/det_keypoint_unite_infer.py", line 339, in main
    topdown_unite_predict_video(detector, topdown_keypoint_detector,
  File "/data/lujunda/drown/code/PaddleDetection-2.5.0/deploy/python/det_keypoint_unite_infer.py", line 189, in topdown_unite_predict_video
    index, keypoint_res['bbox'],
KeyError: 'bbox'


但是这条命令不会
python deploy/python/det_keypoint_unite_infer.py \
--det_model_dir=output_inference/mot_ppyoloe_l_36e_pipeline/ \
--keypoint_model_dir=output_inference/dark_hrnet_w32_256x192 \
--video_file=../work/clip/0619_12.mp4 \
--device=GPU --save_res=True
```

### 训练过程
参数量很少，就是einsum矩阵乘法和一些conv2d堆叠    

```
data = data_batch[0]
label = data_batch[1:]

feature = self.backbone(data)
cls_score = self.head(feature)

bs 1
label[1, 1]

backbone是st-gcn，输入[1, 2, 50, 17, 1]
输出[1, 256, 1, 1]
head: conv变通道，在reshape N,C,1,1 --> N,C
得到[1, 2]

然后计算label和score的cross_entropy
softmax_cross_entropy
两者维度不同如何在底层计算交叉熵？？？，C语言实现无法查看

优化器采用Momentum

维度 大小 说明
N 不定 训练数据集序列数量
C 2 关键点坐标维度，即步骤3中得到的骨骼点坐标(x，y)
T 50 动作序列的时间维度，例如取50帧作为一个动作序列
V 17 每个行人的关键点个数
M 1 人物个数，即每个动作序列只针对单人预测
```
backbone过程
```
x转换
N, C, T, V, M [1, 2, 50, 17, 1]
N, M, V, C, T
N * M, V * C, T [1, 34, 50]
batchnorm
N, M, C, T, V
N * M, C, T, V [1, 2, 50, 17]
进入网络
十个st_gcn_block和edge_importance，每个st_gcn_block含有gcn和tcn
edge_importance边界重要程度edge attention，[3,17,17],'stgcn_0.w_0'
STGCN()
for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
    x, _ = gcn(x, paddle.multiply(self.A, importance))
x = self.pool(x)  # NM,C,T,V --> NM,C,1,1
C = x.shape[1]
x = paddle.reshape(x, (N, M, C, 1, 1)).mean(axis=1)  # N,C,1,1
return x

st-gcn内部
def forward(self, x, A):
        res = self.residual(x)   第一个block是res=0后面是iden或者指定的conv
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A

因为init定义
# build networks
        spatial_kernel_size = A.shape[0]
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        决定一些cfg中没有的定义，决定卷积大小和步长,pad等
        self.data_bn = nn.BatchNorm1D(in_channels *
                                      A.shape[1]) if self.data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.LayerList((
            st_gcn_block(in_channels,
                         64,
                         kernel_size,
                         1,
                         residual=False,
                         **kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, 2, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        ))






(0): st_gcn_block(
      (gcn): ConvTemporalGraphical(
        (conv): Conv2D(2, 192, kernel_size=[1, 1], padding=(0, 0), data_format=NCHW)    
      )
      (tcn): Sequential(
        (0): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
        (1): ReLU()
        (2): Conv2D(64, 64, kernel_size=[9, 1], padding=(4, 0), data_format=NCHW)
        (3): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
        (4): Dropout(p=0, axis=None, mode=upscale_in_train)
      )
      (relu): ReLU()
    )

```


ST‑GCN是一个基于骨骼点坐标序列进行预测的模型，它通过将图卷积网络(GCN)
和时间卷积网络(TCN)结合起来，扩展到时空图模型，设计出了用于行为识别的骨骼点序列
通用表示，该模型将人体骨骼表示为图，其中图的每个节点对应于人体的一个关节点。图中
存在两种类型的边，即符合关节的自然连接的空间边(spatial edge)和在连续的时间步骤
中连接相同关节的时间边(temporal edge)。在此基础上构建多层的时空图卷积，它允许信
息沿着空间和时间两个维度进行整合。

```

gcn图卷积，关节的自然连接的空间边(spatial edge)
ConvTemporalGraphical
这一块的作用仅在于与A边注意力做矩阵乘法。
def forward(self, x, A):
        assert A.shape[0] == self.kernel_size   3

        x = self.conv(x)   [1, 192, 50, 17]主要是变通道,1*1卷积
        n, kc, t, v = x.shape
        x = x.reshape((n, self.kernel_size, kc // self.kernel_size, t, v))   [1, 3, 64, 50, 17]    A[3, 17, 17]
        x = einsum(x, A) 矩阵乘法

        return x, A


爱因斯坦求和约定
爱因斯坦求和约定（einsum）提供了一套既简洁又优雅的规则，可实现包括但不限于：向量内积，向量外积，矩阵乘法，转置和张量收缩（tensor contraction）等张量操作，熟练运用 einsum 可以很方便的实现复杂的张量操作，而且不容易出错。

x = x.transpose((0, 2, 3, 1, 4))
n, c, t, k, v = x.shape  [1, 64, 50, 3, 17]
k2, v2, w = A.shape [3, 17, 17]
要求kv对应相等
x = x.reshape((n, c, t, k * v))
A = A.reshape((k * v, w))
y = paddle.matmul(x, A)
[1, 64, 50, 17]

```
```
时间卷积网络(TCN)
这一块的左右主要在50这个维度，[1, 64, 50, 17]，conv操作，50是
T 50 动作序列的时间维度，例如取50帧作为一个动作序列   
tcn中依次经过
(tcn): Sequential(
        (0): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
        (1): ReLU()
        (2): Conv2D(64, 64, kernel_size=[9, 1], padding=(4, 0), data_format=NCHW)
        (3): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
        (4): Dropout(p=0.5, axis=None, mode=upscale_in_train)
      )
[1, 64, 50, 17]
```
```
10个st-gcn-block结束后
经过pool NM,C,T,V --> NM,C,1,1
将
T 50 动作序列的时间维度，例如取50帧作为一个动作序列
V	17	每个人物关键点的个数
归为1
```

## ST-GCN的技术延展-动作生成

基于对ST-GCN在人体动作识别上的效果,我们将ST-GCN网络与VAE网络结合。目的在于获取人体动作的语义,进而生成人体的动作,最终可以应用于机器人动作模仿或者其他强化学习项目中。

## st-gcn专利日记
2024.1.7晚，可写的，检测跟踪关键点的详细网络结构，网络搭建，st-gcn的详细网络结构  
有点多   
不写了，写怎么应用吧     
使用例子截图，数据截图，   
