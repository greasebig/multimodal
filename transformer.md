# transformer  
  
  Transformer的知名应用——BERT——无监督的训练的Transformer  
  
  ChatGPT, Chat Generative Pre-training Transformer
## Transformer 整体结构
  
  下图是 Transformer 用于中英文翻译的整体结构  

  ![Alt text](assets_picture/transformer/image.png)  

  Encoder 和 Decoder 都包含 6 个 block
## Transformer 的工作流程
  
  第一步：获取输入句子的每一个单词的表示向量 X，X由单词的 Embedding（Embedding就是从原始数据提取出来的Feature） 和单词位置的 Embedding 相加得到  

  ![Alt text](assets_picture/transformer/image-1.png)


  第二步：将得到的单词表示向量矩阵 (如上图所示，每一行是一个单词的表示 x) 传入 Encoder 中，经过 6 个 Encoder block 后可以得到句子所有单词的编码信息矩阵 C，如下图。单词向量矩阵用??表示， n 是句子中单词个数，d 是表示向量的维度 (论文中 d=512)。每一个 Encoder block 输出的矩阵维度与输入完全一致。  

  ![Alt text](assets_picture/transformer/image-2.png)

  第三步：将 Encoder 输出的编码信息矩阵 C传递到 Decoder 中，Decoder 依次会根据当前翻译过的单词 1~ i 翻译下一个单词 i+1，如下图所示。在使用的过程中，翻译到单词 i+1 的时候需要通过 Mask (掩盖) 操作遮盖住 i+1 之后的单词  

  ![Alt text](assets_picture/transformer/image-3.png)  

  Decoder 接收了 Encoder 的编码矩阵 C，然后首先输入一个翻译开始符 "<Begin>"，预测第一个单词 "I"；然后输入翻译开始符 "<Begin>" 和单词 "I"，预测单词 "have"，以此类推。
## Transformer 的输入
  
### 单词 Embedding
  单词的 Embedding 有很多种方式可以获取，例如可以采用 Word2Vec、Glove 等算法预训练得到，也可以在 Transformer 中训练得到。
### 位置 Embedding  
  
  Transformer  中除了单词的 Embedding，还需要使用位置 Embedding 表示单词出现在句子中的位置。**因为 Transformer 不采用 RNN 的结构，而是使用全局信息，不能利用单词的顺序信息**，而这部分信息对于 NLP 来说非常重要。所以 Transformer 中使用位置 Embedding 保存**单词在序列中的相对或绝对位置**。

  PE 可以通过训练得到，也可以使用某种公式计算得到。在 Transformer 中采用了后者，计算公式如下：  

  ![Alt text](assets_picture/transformer/image-4.png)  

  其中，pos 表示单词在句子中的位置，d 表示 PE的维度 (与词 Embedding 一样)，2i 表示偶数的维度，2i+1 表示奇数维度 (即 2i≤d, 2i+1≤d)。使用这种公式计算 PE 有以下的好处：
  - 使 PE 能够**适应比训练集里面所有句子更长的句子**，假设训练集里面最长的句子是有 20 个单词，突然来了一个长度为 21 的句子，则使用公式计算的方法可以计算出第 21 位的 Embedding。
  - 可以让模型**容易地计算出相对位置**，对于固定长度的间距 k，PE(pos+k) 可以用 PE(pos) 计算得到。因为 Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)。  

  将单词的词 Embedding 和位置 Embedding 相加，就可以得到单词的表示向量 x，x 就是 Transformer 的输入。

## Self-Attention（自注意力机制）
  
  论文中 Transformer 的内部结构图，左侧为 Encoder block，右侧为 Decoder block。红色圈中的部分为 Multi-Head Attention，是由多个 Self-Attention组成的，可以看到 Encoder block 包含一个 Multi-Head Attention，而 Decoder block 包含两个 Multi-Head Attention (其中有一个用到 Masked)。Multi-Head Attention 上方还包括一个 Add & Norm 层，Add 表示残差连接 (Residual Connection) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。  

  block  

  ![Alt text](assets_picture/transformer/image-5.png)

### Self-Attention 结构  
  

  ![Alt text](assets_picture/transformer/image-6.png)  

  在计算的时候需要用到矩阵Q(查询),K(键值),V(值)。在实际中，Self-Attention 接收的是输入(单词的表示向量x组成的矩阵X) 或者上一个 Encoder block 的输出。而**Q,K,V正是通过 Self-Attention 的输入进行线性变换得到的**，如下

### Q, K, V 的计算
  
  **Self-Attention 的输入**用矩阵X进行表示，则可以**使用线性变阵矩阵WQ,WK,WV计算得到Q,K,V**。计算如下图所示，注意 X, Q, K, V 的每一行都表示一个单词。  
  ![Alt text](assets_picture/transformer/image-7.png)

### Self-Attention 的输出
  
  得到矩阵 Q, K, V之后就可以计算出 Self-Attention 的输出了，***计算公式***如下：  
  ![Alt text](assets_picture/transformer/image-8.png)  

  公式中计算矩阵Q和K每一行向量的内积  
  Q乘以K的转置后，得到的矩阵行列数都为 n，n 为句子单词数，这个矩阵可以表示单词之间的 attention 强度。下图为Q乘以 
 ，1234 表示的是句子中的单词。  
 ![Alt text](assets_picture/transformer/image-9.png)  
 得到之后，使用 Softmax 计算每一个单词对于其他单词的 attention 系数，公式中的 Softmax 是对矩阵的每一行进行 Softmax，即每一行的和都变为 1.   

 ![Alt text](assets_picture/transformer/image-10.png)  

 （Logit：通常用sigmoid函数表示，例如sigmoid(x) = 1 / (1 + exp(-x))。  

Softmax：Softmax函数的公式是exp(xi) / Σ(exp(xj))，其中xi是输入向量中的元素，Σ表示对所有元素求和。  

在深度学习中，通常是通过计算logit，然后通过Softmax函数将logit转换为概率分布，以用于多分类问题。）  

得到 Softmax 矩阵之后可以和V相乘，得到最终的输出Z。  
![Alt text](assets_picture/transformer/image-11.png)  

上图中 Softmax 矩阵的第 1 行表示单词 1 与其他所有单词的 attention 系数，最终单词 1 的输出 
 等于所有单词 i 的值 
 根据 attention 系数的比例加在一起得到，如下图所示：  

![Alt text](assets_picture/transformer/image-12.png)

## Multi-Head Attention  
  
  ![Alt text](assets_picture/transformer/image-13.png)  
  首先将输入X分别传递到 h 个不同的 Self-Attention 中，计算得到 h 个输出矩阵Z  
  ![Alt text](assets_picture/transformer/image-14.png)  
  得到 8 个输出矩阵 
 到 
 之后，Multi-Head Attention 将它们拼接在一起 (Concat)，然后传入一个Linear层，得到 Multi-Head Attention 最终的输出Z  
 Multi-Head Attention 输出的矩阵Z与其输入的矩阵X的维度是一样的  

## Encoder 结构