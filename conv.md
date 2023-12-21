# 计算

## conv计算
输出大小  
![Alt text](assets_picture/conv/image.png)    
![Alt text](assets_picture/conv/image-1.png)    

计算公式  
torch.nn.Conv2d   
$$
\text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
\sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
$$


where $`\star`$ is the valid 2D `cross-correlation`_ operator,  
    $ `N` $ is a batch size,   
    $`C`$ denotes a number of channels,   
    $`H` $ is a height of input planes in pixels,   
    and $`W`$ is
    width in pixels.
$$

              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
$$
$$

              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
$$
Attributes:  

weight (Tensor): the learnable weights of the module of shape
            $$`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            `\text{kernel\_size[0]}, \text{kernel\_size[1]})`.$$
            The values of these weights are sampled from
            $$`\mathcal{U}(-\sqrt{k}, \sqrt{k})` $$where
            $$`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`$$
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from $$`\mathcal{U}(-\sqrt{k}, \sqrt{k})` $$where
           $$`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`$$

dilation = 2  
![Alt text](assets_picture/conv/image-2.png)   


## GroupNorm
$$ y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta $$   


## SiLU
 优点：计算速度比ReLU函数更快，因为它只涉及一个sigmoid函数的计算；Silu函数在接近0时的导数接近1，能够保留更多的信息。

Silu的缺点：Silu函数在接近正无穷和负无穷时的导数接近0，可能导致梯度消失问题；Silu函数的值域在(0,1)之间，可能会导致信息的损失。
  
$$  \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.} $$  
![Alt text](assets_picture/conv/image-3.png)  
SiLU是Sigmoid和ReLU的改进版。SiLU具备无上界有下界、平滑、非单调的特性。SiLU在深层模型上的效果优于 ReLU。可以看做是平滑的ReLU激活函数。

SiLU（Sigmoid Linear Unit）激活函数也被称为 Swish 激活函数，它是 Google Brain 在 2017 年引入的一种自适应激活函数。

Swish 函数的定义如下：
f(x) = x * sigmoid(x)


## pad
Padding size:
The padding size by which to pad some dimensions of :attr:`input`
    are described starting from the last dimension and moving forward.
    $$`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor`$$ dimensions
    of ``input`` will be padded.
    For example,   
    to pad only the last dimension of the input tensor, then
    :attr:`pad` has the form
    :$$ :`(\text{padding\_left}, \text{padding\_right})` $$;
    to pad the last 2 dimensions of the input tensor, then use
    :$$:`(\text{padding\_left}, \text{padding\_right},`
    :`\text{padding\_top}, \text{padding\_bottom})`$$;
    to pad the last 3 dimensions, use
    :$$:`(\text{padding\_left}, \text{padding\_right},`
    ::`\text{padding\_top}, \text{padding\_bottom}`
    ::`\text{padding\_front}, \text{padding\_back})`.$$
pad (tuple): m-elements tuple, where
        :$$:`\frac{m}{2} \leq`  input dimensions $$and :$:`m`$ is even.  
从后往前，两个一组  
Examples::

    >>> t4d = torch.empty(3, 3, 4, 2)
    >>> p1d = (1, 1) # pad last dim by 1 on each side
    >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
    >>> print(out.size())
    torch.Size([3, 3, 4, 4])
    >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
    >>> out = F.pad(t4d, p2d, "constant", 0)
    >>> print(out.size())
    torch.Size([3, 3, 8, 4])
    >>> t4d = torch.empty(3, 3, 4, 2)
    >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
    >>> out = F.pad(t4d, p3d, "constant", 0)
    >>> print(out.size())
    torch.Size([3, 9, 7, 3])




## GeGLU 
is an activation function which is a variant of GLU. The definition is as follows:  
$\text{GeGLU}\left(x, W, V, b, c\right) = \text{GELU}\left(xW + b\right) \otimes \left(xV + c\right)$  
GLU: Gated Linear Unit   
GLU通过门控机制对输出进行把控，像Attention一样可看作是对重要特征的选择。其优势是不仅具有通用激活函数的非线性，而且反向传播梯度时具有线性通道，类似ResNet残差网络中的加和操作传递梯度，能够缓解梯度消失问题。    
A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.   
```
hidden_states, gate = self.proj(hidden_states, *args).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)
分别赋值给hidden_states和gate。这在很多门控循环神经网络（Gated Recurrent Neural Networks，如LSTM和GRU）中是常见的操作，其中门控机制用于控制信息的流动。
```

### GELU 高斯误差线性单元
优点：  
似乎是 NLP 领域的当前最佳；尤其在 Transformer 模型中表现最好  
能避免梯度消失问题。

优点：激活函数的值域在整个实数范围内，避免了sigmoid函数在极端值处的梯度消失问题； 激活函数的导数在大部分区间内都为非零值，避免了ReLU函数在负数区间内的梯度为0问题；Gelu函数在接近0时的导数接近1，能够保留更多的信息。

缺点：Gelu函数的计算比ReLU函数复杂，计算速度较慢；Gelu函数在负数区间内仍然存在梯度消失问题。


激活函数GELU的灵感来源于 relu 和 dropout，在激活中引入了随机正则的思想。gelu通过输入自身的概率分布情况，决定抛弃还是保留当前的神经元。  
![Alt text](assets_picture/conv/image-4.png)  
![Alt text](assets_picture/conv/image-5.png)   
可以理解为，对于输入的值，根据它的情况乘上 1 或 0。更「数学」一点的描述是，对于每一个输入 x，其服从于标准正态分布 N(0, 1)，它会乘上一个伯努利分布 Bernoulli(Φ(x))，其中Φ(x) = P(X ≤ x)。  
随着 x 的降低，它被归零的概率会升高。对于 ReLU 来说，这个界限就是 0，输入少于零就会被归零。这一类激活函数，不仅保留了概率性，同时也保留了对输入的依赖性。  

gelu在最近的Transformer模型中（包括BERT，RoBertA和GPT2等）得到了广泛的应用。

    

## sigmoid
主要优点：  
函数的映射范围是 0 到 1，对每个神经元的输出进行了归一化  
梯度平滑，避免「跳跃」的输出值  
函数是可微的，意味着可以找到任意两个点的   sigmoid 曲线的斜率  
预测结果明确，即非常接近 1 或 0  

缺点：  
倾向于梯度消失  
函数输出不是以 0 为中心的，会降低权重更新的效率  
Sigmoid 函数执行指数运算，计算机运行得较慢 

sigmoid 是最基础的激活函数，可以将任意数值转换为概率（缩放到0～1之间）.在分类等场景中有广泛的应用。  
![Alt text](assets_picture/conv/image-6.png)  
![Alt text](assets_picture/conv/image-7.png)  
 

## Tanh 双曲正切激活函数
相比sigmoid，tanh的优势在于：  
tanh 的输出间隔为 1，并且整个函数以 0 为中心，比 sigmoid 函数更好  
在负输入将被强映射为负，而零输入被映射为接近零。  

缺点：  
当输入较大或较小时，输出几乎是平滑的并且梯度较小，这不利于权重更新


激活函数Tanh和sigmoid类似，都是 S 形曲线，输出范围是[-1, 1]。  
在一般的二元分类问题中，tanh 函数常用于隐藏层，sigmoid 用于输出层，但这并不是固定的，需要根据特定问题进行调整。   
![Alt text](assets_picture/conv/image-8.png)  
![Alt text](assets_picture/conv/image-9.png)    

## ReLU 整流线性单元
相比sigmoid和tanh，它具有以下优点：  
当输入为正时，不存在梯度饱和问题  
计算复杂度低。ReLU 函数只存在线性关系，一个阈值就可以得到激活值  
单侧抑制，可以对神经元进行筛选，让模型训练更加鲁棒  

当然它也存在缺点：  
dead relu 问题（神经元坏死现象）。relu在训练的时很“脆弱”。在x<0时，梯度为0，这个神经元及之后的神经元梯度永远为0，不再对任何数据有所响应，导致相应参数永远不会被更新  
解决方法：采用Xavier初始化方法，以及避免将learning rate设置太大或使用adagrad等自动调节learning rate的算法。  
输出不是 0 均值    

![Alt text](assets_picture/conv/image-10.png)  
Relu也是深度学习中非常流行的激活函数，近两年大火的Transformer模块由Attention和前馈神经网络FFN组成，其中FFN(即全连接)又有两层，第一层的激活函数就是ReLU，第二层是一个线性激活函数。    


## AdamW
AdamW优化器修正了Adam中权重衰减的bug    
AdamW与Adam对比，主要是修改了权重衰减计算的方式，一上来直接修改了 
theta_t ，而不是把权重衰减放到梯度里，由梯度更新间接缩小 theta_t   


## Adam
![Alt text](assets_picture/conv/image-11.png)  

