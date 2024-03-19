# transformer  
怎么手写代码公式？？？？       

  2017  
  
  Transformer的知名应用——BERT——无监督的训练的Transformer  

  ChatGPT, Chat Generative Pre-training Transformer  

  ![Alt text](assets_picture/transformer/image-23.png)    
![Alt text](assets_picture/transformer/image-13.png)  
![Alt text](assets_picture/transformer/image-6.png) 
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
  对于较大的dk
来说在完成qkt
后将会得到很大的值，而这将导致在经过sofrmax操作后产生非常小的梯度，不利于网络的训练。  

  公式中计算矩阵Q和K每一行向量的内积  
  Q乘以K的转置后，得到的矩阵行列数都为 n，n 为句子单词数，这个矩阵可以表示单词之间的 attention 强度。下图为Q乘以 KT
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
  ![Alt text](assets_picture/transformer/image-15.png)

### Add & Norm

  Add指 X+MultiHeadAttention(X)，是一种残差连接，通常用于解决多层网络训练的问题，可以让网络**只关注当前差异的部分**，在 ResNet 中经常用到  
  Norm指 Layer Normalization，通常用于 RNN 结构，Layer Normalization 会将每一层神经元的输入都转成均值方差都一样的，这样可以加快收敛。
### Feed Forward
  
  Feed Forward 层比较简单，是一个`两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数`，对应的公式如下  
  ![Alt text](assets_picture/transformer/image-16.png)  
  X是输入，Feed Forward 最终得到的输出矩阵的维度与X一致。  

  - 激活函数  

  ![Alt text](assets_picture/transformer/image-17.png)  
  如果不用激活函数，在这种情况下每一层输出都是上层输入的线性函数。容易验证，无论神经网络有多少层，输出都是输入的线性组合，与没有隐藏层效果相当，这种情况就是最原始的感知机（Perceptron）了。  
  （不再是输入的线性组合，可以逼近任意函数）。最早的想法是sigmoid函数或者tanh函数，输出有界，很容易充当下一层输入。  
  
  引入ReLu的原因

第一，**计算量**。采用sigmoid等函数，算激活函数时（指数运算），计算量大，反向传播求误差梯度时，求导涉及除法，计算量相对大，而采用Relu激活函数，整个过程的计算量节省很多。

第二，***梯度消失**。对于深层网络，sigmoid函数反向传播时，很容易就会出现 梯度消失 的情况（在sigmoid接近**饱和区时，变换太缓慢，导数趋于0**，这种情况会造成信息丢失），从而无法完成深层网络的训练。

第三，**稀疏性**。ReLu会使一部分神经元的输出为0，这样就造成了 网络的稀疏性，并且减少了参数的相互依存关系，缓解了**过拟合**问题的发生。

## Decoder 结构

  与 Encoder block 相似，但是存在一些区别：

- 包含两个 Multi-Head Attention 层。
- 第一个 Multi-Head Attention 层采用了 Masked 操作。
- 第二个 Multi-Head Attention 层的**K, V**矩阵使用 Encoder 的**编码信息矩阵C**进行计算，而**Q**使用上一个 **Decoder block 的输出**计算。
- 最后有一个 Softmax 层计算下一个翻译单词的概率。

`在unet交叉注意力层也是如此，kv来自condition, q来自上一个block`


###  第一个 Multi-Head Attention
  
  首先根据输入 "'<Begin'>" 预测出第一个单词为 "I"，然后根据输入 "'<Begin'> I" 预测下一个单词 "have"。  
  ![Alt text](assets_picture/transformer/image-19.png)  

  Decoder 可以在训练的过程中使用 Teacher Forcing 并且**并行化训练**，即将正确的单词序列 ('<Begin'> I have a cat) 和对应输出 (I have a cat '<end'>) 传递到 Decoder。  
  那么在预测第 i 个输出时，就要将第 i+1 之后的单词掩盖住，注意 **Mask 操作是在 Self-Attention 的 Softmax 之前使用**的，下面用 0 1 2 3 4 5 分别表示 "'<Begin'> I have a cat '<end'>"。  
  ![Alt text](assets_picture/transformer/image-6.png)  

  - 第一步：是 Decoder 的输入矩阵和 Mask 矩阵，输入矩阵包含 "'<Begin'> I have a cat" (0, 1, 2, 3, 4) 五个单词的表示向量，Mask 是一个 5×5 的矩阵。在 Mask 可以发现单词 0 只能使用单词 0 的信息，而单词 1 可以使用单词 0, 1 的信息，即只能使用之前的信息。  
  ![Alt text](assets_picture/transformer/image-18.png)  

  - 第二步：接下来的操作和之前的 Self-Attention 一样，通过输入矩阵X计算得到Q,K,V矩阵。然后计算Q和 KT
 的乘积 
 。  
 ![Alt text](assets_picture/transformer/image-20.png)
 - 第三步：得到 Mask QKT
 之后在 Mask QKT
上进行 Softmax，每一行的和都为 1。但是单词 0 在单词 1, 2, 3, 4 上的 attention score 都为 0。  
![Alt text](assets_picture/transformer/image-21.png)  
- 第四步：使用 Mask QKT
与矩阵 V相乘，得到输出 Z，则单词 1 的输出向量 Z1
 是只包含单词 1 信息的。  
 ![Alt text](assets_picture/transformer/image-22.png)

- 第五步：通过上述步骤就可以得到一个 Mask Self-Attention 的输出矩阵 
 ，然后和 Encoder 类似，通过 Multi-Head Attention 拼接多个输出
 然后计算得到第一个 Multi-Head Attention 的输出Z，Z与输入X维度一样。

### 第二个 Multi-Head Attention
  
  根据 Encoder 的输出 C计算得到 K, V，根据上一个 Decoder block 的输出 Z 计算 Q (如果是第一个 Decoder block 则使用输入矩阵 X 进行计算)  
  这样做的好处是在 Decoder 的时候，每一位单词都可以利用到 Encoder 所有单词的信息 (这些信息无需 Mask)。??

### Softmax 预测输出单词
![Alt text](assets_picture/transformer/image-23.png)  
  Decoder block 最后的部分是利用 Softmax 预测下一个单词，在之前的网络层我们可以得到一个最终的输出 Z，因为 Mask 的存在，使得单词 0 的输出 Z0 只包含单词 0 的信息，如下：  
  ![Alt text](assets_picture/transformer/image-24.png)  
  ![Alt text](assets_picture/transformer/image-25.png)  
  Softmax 根据输出矩阵的每一行预测下一个单词    
  与 Encoder 一样，Decoder 是由多个 Decoder block 组合而成

## Transformer 总结
  
  - Transformer 与 RNN 不同，可以比较好地并行训练。
- Transformer 本身是不能利用单词的顺序信息的，因此需要在输入中添加位置 Embedding，否则 Transformer 就是一个词袋模型了。
- Transformer 的重点是 Self-Attention 结构，其中用到的 Q, K, V矩阵通过输出进行线性变换得到。
- Transformer 中 Multi-Head Attention 中有多个 Self-Attention，可以捕获单词之间多种维度上的相关系数 attention score。  
![Alt text](assets_picture/transformer/image-23.png)    
![Alt text](assets_picture/transformer/image-13.png)  
![Alt text](assets_picture/transformer/image-6.png)  

## 运行逻辑
  
  - 训练时：第i个decoder的输入 = encoder输出 + ground truth embeding
  - 预测时：第i个decoder的输入 = encoder输出 + 第(i-1)个decoder输出

  训练时因为知道ground truth embeding，相当于知道正确答案，网络可以一次训练完成。  
  
预测时，首先输入start，输出预测的第一个单词 然后start和新单词组成新的query，再输入decoder来预测下一个单词，循环往复 直至end

## QA
### 1.为什么要shifted right
整体右移一位  
Shifted Right 实质上是给输出添加起始符/结束符，方便预测第一个Token/结束预测过程。   
### 2.多头注意力，本质就是拆开自注意力要计算的张量去分开计算，然后算qk分数和qkv最后分数，有什么用？
多头注意力例子  

residual,残差2  
  不做prepare-attn-mask      
  toq,tok,tov   
  8个头，to_qkv后做head_to_batch_dim：Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.    
  If output_dim=`3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.  
  view(batch_size * num_heads, -1, dim_per_head)   
  torch.Size([4, 4096, 320])变torch.Size([32, 4096, 40])   
  计算score : torch.Size([32, 4096, 4096])。`多个头确实score第一维度更多八倍`   
  计算qkv结果，即selfattn结果torch.Size([32, 4096, 40])`qkv结果数量一样`  
  batch_to_head_dim ：torch.Size([4, 4096, 320])  
  linear,drop(0)  
  加残差2    

表达能力： 多头注意力使得模型可以学习多个不同的关注点或表示空间。每个注意力头都可以专注于学习数据中的不同关系或特征，从而提高模型对复杂关系的建模能力  
降低过拟合： 多头注意力可以被视为一种正则化机制，因为它允许模型通过关注不同的信息源来减轻过拟合的风险。每个头都相当于模型中的一个子模型，可以减小过拟合的可能性。  

论文作者提出用于克服「模型在对当前位置的信息进行编码时，会过度的将注意力集中于自身的位置」的问题。   
原论文中说的是，将模型分为多个头，形成多个子空间，可以让模型去关注不同方面的信息   

## 代码实现
```python
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)      # [N, seq_len, d_model]

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)   # [N, head, seq_len, d_model]

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)     # out:[N, head, seq_len, d_model]

        # 4. concat and pass to linear layer
        out = self.concat(out)          # [N, seq_len, d_model]
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

```



