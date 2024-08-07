# CLIP
结构        
![Alt text](assets_picture/clip/image-1.png)


2021
- 论文中总结的不足
![image](https://github.com/greasebig/multimodal/assets/121388156/85fa96b0-b9c8-4e4c-87b2-fa774557f0a7)
- CLIP是如何进行预训练的？      
对比学习
- CLIP是如何做zero-shot的推理的？
```
prompt template。比如对于ImageNet的类别，首先把它变成"A photo of a {object}" 这样一个句子
拿图片的特征和1000个文本特征算余弦相似性
直接用类别单词去抽取文本特征也可以，但是模型预训练的时候和图片配对的都是句子，推理的时候用单词效果会下降

1.只用一个单词去做prompt，会经常出现歧义性的问题
2.匹配的文本一般都是一个句子，很少出现一个单词的情况，如果推理的时候，每次进来的是一个单词，可能就存在distribution gap的问题，抽出来的特征可能就不好
```
- 数据集构造方式（未读懂）
```
400 million的图像文本对，这个数据集称作WIT（WebImage Text）
```
  
首先使用在英文维基百科中出现了超过 100 次的单词构建了50万个queries,            
每个样本的 text 要至少包含这 50 万个 queries 中的一个,      
每个 query 最多有 2 万个图像-文本对,     
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
encoder后再经过线性映射统一到同一维度，计算余弦相似性，loss采用交叉熵损失label分别对余弦矩阵两个维度计算，loss取平均          
![image](https://github.com/greasebig/multimodal/assets/121388156/a1536d43-64c1-4c98-bb81-a974556105ca)      
```得到对应的特征之后，再经过一个投射层（即W_i和W_t)，投射层的意义是学习如何从单模态变成多模态，
投射完之后再做L2 norm，就得到了最终的用来对比的特征I_e和T_e，

现在有n个图像的特征，和n个文本的特征，接下来就是算consine similarity，
算的相似度就是最后要分类的logits，最后logits和ground truth做交叉熵loss
```

- 训练

  从头开始训练，文本和图片的encoder都不需要使用预训练的weights，between the representation and the constastive embedding space也没有使用非线性的投射（projection），use only a linear projection to map from each encoder's representation to the multi-modal embedding space. 在之前对比学习的一些文章中提到过，非线性投射层比线性投射层能够带来将近10个点的性能提升，但是在CLIP中，作者发现线性还是非线性关系不大，他们怀疑非线性的投射层是用来适配纯图片的单模态学习的。也不需要做太多的数据增强，唯一用的是随机裁剪（a random square crop from resized images）  

  简化数据增强：仅使用随机裁剪



- Choosing and Scaling a Model
```
image encoder选了ResNet和ViT两种结构，text encoder只用了transformer
```
  
  用ResNet-50作为base architecture，然后又对原始版本做了一些改动，利用attention pooling mechanism代替了global average pooling  
  使用的ViT，只做了一点很小的修改，add an additional layer normalization to the combined patch and position embeddings before the transformer 并且使用了一个略微不同的初始化方案  

  Text encoder是一个transformer，使用一个63M-parameter 12-layer 512-wide model with 8 attention heads作为base size  
  序列长度最大为76  

  将 text encoder/image encoder 输出的 feature embedding 再过一层**线性投影**而非线性投影（常见模型如simCLR 是用了非线性投影，但是作者观察不到线性/非线性投影的区别，并认为在自监督表示学习方法中，非线性投影才能与当前图像的细节相适应）

- Training  

  图片这边共训练了8个模型，5个ResNet和3个transformer，5个ResNet包括ResNet-50，ResNet-101，另外三个是根据efficientNet的方式对ResNet-50的宽度、深度、输入大小进行scale，分别对应原始ResNet50 4倍，16倍，64倍的计算量，3个transformer包括ViT-B/32，ViT-B/16 和ViT-L/14。所有的模型都训练了32个epoch，用的adam优化器  

  超参数是基于grid searches，random search和manual tuning来调整的，为了让调参更快，超参搜索的时候是用的Res50，只训练了一个epoch。batch size 32768  

  最大的那个ResNet（RN50x64）在592个V100的GPU上训练了18天，最大的ViT在256个V100 GPU上只花了12天。对预训练好的ViT-L/14，又在这个数据集上fine-tune了一个epoch，用的是更大尺寸（336*336的），这个模型称作ViT-L/14@336px

- Zero-Shot Transfer  

  借助文本训练了一个又大又好的模型之后，就可以借助这个文本作为引导，很灵活的做zero-shot的迁移学习。
- prompt ensembling
  
  集成多个zero shot classifiers，即prompt ensembling ，作为提高性能的另一种方式。这些分类器是在不同的上下文提示下得到的，比如“A photo of a big {label}" 和”A photo of a small {label}"。  

  ![Alt text](assets_picture/clip/image.png)  
  
  列出了使用的这80个context prompts.在ImageNet上，共集成了80个不同的context prompts，这比单个的default prompt 提高了3.5%的性能。
## CLIP拓展应用：DALL-E 与 DALL-E2
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
DALL·E中的Transformer结构由64层attention层组成，每层的注意力头数为62，每个注意力头的维度为64，因此，每个token的向量表示维度为62*64=3968。如图所示，attention层使用了行注意力mask、列注意力mask和卷积注意力mask三种稀疏注意力。
### VAE基本原理
在AntoEncoder的基础上给lantent vector添加限制条件，让其服从高斯分布，这样我们通过训练得到的decoder就可以直接使用，将随机生成的一个高斯分布喂给decoder就能生成图片，如上面第一张图所示。



### DALL-E2
- 分辨率达到1024，是4倍dall-e
- 基本原理是 CLIP + DDPM
- DALL-E 2中的先验子模型和图像生成子模型都是基于扩散模型的
- DALL-E 2使用了一种改进的GLIDE模型，以两种方式使用投影的CLIP文本嵌入(# 未读懂)

## CLIP拓展应用 三篇CVPR2022
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

