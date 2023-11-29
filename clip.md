# CLIP 及其相关
- 论文中总结的不足
![image](https://github.com/greasebig/multimodal/assets/121388156/85fa96b0-b9c8-4e4c-87b2-fa774557f0a7)
- CLIP是如何进行预训练的？
对比学习
- CLIP是如何做zero-shot的推理的？
```
prompt template。比如对于ImageNet的类别，首先把它变成"A photo of a {object}" 这样一个句子
拿图片的特征和1000个文本特征算余弦相似性
直接用类别单词去抽取文本特征也可以，但是模型预训练的时候和图片配对的都是句子，推理的时候用单词效果会下降
```
- 数据集构造方式（未读懂）
```
400 million的图像文本对，这个数据集称作WIT（WebImage Text）
```
- Efficient Pre-Training Method
```
试图预测每张图片所附文本的确切单词（给定一张图片，去预测对应文本，要逐字逐句去预测文本的），这是个困难的任务
探索训练了一个系统来解决可能更容易的任务，即只预测哪个文本作为一个整体和哪个图像配对，而不是预测该文本的确切单词
```
- loss
```
算两个loss，一个是image的，一个是text的，最后把两个loss加起来就平均。
这个操作在对比学习中是很常见的，都是用的这种对称式的目标函数
```
- CLIP核心实现的伪代码
![image](https://github.com/greasebig/multimodal/assets/121388156/a1536d43-64c1-4c98-bb81-a974556105ca)
```得到对应的特征之后，再经过一个投射层（即W_i和W_t)，投射层的意义是学习如何从单模态变成多模态，
投射完之后再做l2 norm，就得到了最终的用来对比的特征I_e和T_e，

现在有n个图像的特征，和n个文本的特征，接下来就是算consine similarity，
算的相似度就是最后要分类的logits，最后logits和ground truth做交叉熵loss
```

- 训练
```
  从头开始训练，文本和图片的encoder都不需要使用预训练的weights，between the representation and the constastive embedding space也没有使用非线性的投射（projection），use only a linear projection to map from each encoder's representation to the multi-modal embedding space. 在之前对比学习的一些文章中提到过，非线性投射层比线性投射层能够带来将近10个点的性能提升，但是在CLIP中，作者发现线性还是非线性关系不大，他们怀疑非线性的投射层是用来适配纯图片的单模态学习的。也不需要做太多的数据增强，唯一用的是随机裁剪（a random square crop from resized images）
```
这是一行文本  
这是换行的下一行文本

- Choosing and Scaling a Model
```
将image encoder选了ResNet和ViT两种结构，text encoder只用了transformer
```

## 拓展应用：DALL-E 与 DALL-E2
```
基本原理为 VQGAN + CLIP。
VQGAN（由VAE改进） 相当于生成器，CLIP相当于判别器，
计算文本特征与生成图像特征的相似度（相似表明生成的图像质量高）。
VQGAN 基本原理如下图，先利用特征字典codebook将图像特征离散化（即在codebook中查找最接近的特征，作为图像某一patch的特征），
在decoder阶段利用CLIP计算离散的图像特征与文本特征关系（文本特征本就是离散的word）。
其中，codebook可利用Transformer结构进行监督学习
具体参数是256维文本 token 与 1024维图像 token
```
DALL·E的整体流程
```
1.第一个阶段，先训练一个dVAE(等同于VQGAN),把每张 256x256的RGB图片压缩成32x32的图片token，
每个位置有8192种可能的取值(也就是说dVAE的encoder输出是维度为32x32x8192的logits，
然后通过logits索引codebook(可学习)的特征进行组合)。
```
```
2.第二阶段，用BPE Encoder对文本进行编码，得到256个文本token(不满256的话padding到256)，
然后 将256个文本token与1024个图像token进行拼接，得到长度为1280的数据，最后将拼接的数据输入Transformer中进行自回归训练。
```
```
3.推理阶段，给定一张候选图片和一条文本，通过transformer可以得到融合后的token，
然后用dVAE的decoder生成图片，最后通过预训练好的CLIP计算出文本和生成图片的匹配分数，
采样越多数量的图片，就可以通过CLIP得到不同采样图片的分数排序。
```
- DALL·E中的Transformer结构由64层attention层组成，每层的注意力头数为62，每个注意力头的维度为64，因此，每个token的向量表示维度为62*64=3968。如图所示，attention层使用了行注意力mask、列注意力mask和卷积注意力mask三种稀疏注意力。
### VAE基本原理
在AntoEncoder的基础上给lantent vector添加限制条件，让其服从高斯分布，这样我们通过训练得到的decoder就可以直接使用，将随机生成的一个高斯分布喂给decoder就能生成图片，如上面第一张图所示。



## DALL-E2
- 分辨率达到1024，是4倍dalle
- 基本原理是 CLIP + DDPM
- DALL-E 2中的先验子模型和图像生成子模型都是基于扩散模型的
- DALL-E 2使用了一种改进的GLIDE模型，以两种方式使用投影的CLIP文本嵌入(# 未读懂)

## 拓展应用 三篇CVPR2022
- ActionCLIP ：A new paradigm for Video Action Recognition
![image](https://github.com/greasebig/multimodal/assets/121388156/a6e1ed84-6217-4d2d-95c7-0121d7c23e92)

- CLIP-Event：Connecting Text and Images with Event Structures
将事件中的人与动作链接起来，相当于先通过文本抽取一些关系组合，再与图像进行配对<br>
正样本就是抽取的事件，负样本为替换的其他事件，也可替换动作主体。
- CLIPSeg：Image Segmentation Using Text and Image Prompts
![image](https://github.com/greasebig/multimodal/assets/121388156/2d3c843c-13ae-44d6-bb3e-c550e5c137cb)
- StyleCLIP
- CLIPDraw
- CLIPS

