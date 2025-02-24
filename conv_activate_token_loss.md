# 计算

## conv2d计算
torch.nn.Conv2d(
	in_channels, 
	out_channels, 
	kernel_size, 
	stride=1, 
	padding=0, 
	dilation=1, 
	groups=1, 
	bias=True, 
	padding_mode='zeros', 
	device=None, 
	dtype=None
)          

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


DNN图像分类的问题   
如果直接将图像根据各像素点的向量作为图片特征输入模型，例如LR、SVM、DNN等模型进行分类，理论上可行，但是面临以下问题：      
图像的平移旋转、手写数字笔迹的变化等，会造成输入图像特征矩阵的剧烈变化，影响分类结果。即不抗平移旋转等。     
一般图像像素很高，如果直接DNN这样全连接处理，计算量太大，耗时太长；参数太多需要大量训练样本       

### 卷积的原理       
卷积的本质是有效提取相邻像素间的相关特征，而1×1卷积显然没有此作用。      
1×1卷积不识别空间模式，只融合通道。并常用来改变通道数，降低运算量和参数量。同时增加一次非线性变化，提升网络拟合能力。   

![alt text](assets_picture/conv_activate_token_loss/image-28.png)    
注意相乘的顺序是相反的，这是卷积的定义决定的。   


CNN通过卷积核对图像的各个子区域进行特征提取，而不是直接从像素上提取特征。子区域称为感受野。      
卷积运算：图片感受野的像素值与卷积核的像素值进行按位相乘后求和，加上偏置之后过一个激活函数（一般是Relu）得到特征图Feature Map。       
卷积后都是输出特定形状的强度值，与卷积核形状差异过大的感受野输出为0（经过Relu激活），所以卷积核也叫滤波器Filter。         

使用一个多通道卷积核对多通道图像卷积，结果仍是单通道图像（多通道分别卷积后加和得到最终结果）。要想保持多通道结果，就得使用多个卷积核。     
![alt text](assets_picture/conv_activate_token_loss/image-21.png)      

为什么有效？      
能够对图像提取出想要的信息，能够对图像操作        
![alt text](assets_picture/conv_activate_token_loss/image-22.png)     

对图像的每个像素进行编号，用 x i,j
表示图像的第行第列元素；用 W m,n
表示卷积核filter第m行第n列权重，用 W b
表示filter的偏置项；用 a i,j
表示特征图Feature Map的第i行第j列元素；用 f
表示激活函数(这个例子选择relu函数作为激活函数)。使用下列公式计算卷积：       
![alt text](assets_picture/conv_activate_token_loss/1710588971496.png)     

  
如果卷积前的图像深度为D，那么相应的filter的深度也必须为D。我们扩展一下上式，得到了深度大于1的卷积计算公式：    
D是深度（卷积核个数）      
![alt text](assets_picture/conv_activate_token_loss/1710589034625.png)     


### VGGnet：使用块、小尺寸卷积效果好
VGGnet：使用块、小尺寸卷积效果好     
而多个小尺寸卷积可以达到相同的效果，且参数量更小。还可以多次进行激活操作，提高拟合能力。      
一个5×5卷积参数量25，可以替换成两个3×3卷积。，参数量为18。每个3×3卷积可以替换成3×1卷积加1×3卷积，参数量为12。     


### SENet、CBAM特征通道加权卷积。注意力机制     

SE模块，对各通道中所有数值进行全局平均，此操作称为Squeeze。比如28×28×128的图像，操作后得到128×1的向量。      
此向量输入全连接网络，经过sigmoid输出128维向量，每个维度值域为（0,1），表示各个通道的权重      
在正常卷积中改为各通道加权求和，得到最终结果    
![alt text](assets_picture/conv_activate_token_loss/image-25.png)     
Squeeze建立channel间的依赖关系；Excitation重新校准特征。二者结合强调有用特征抑制无用特征     
能有效提升模型性能，提高准确率。几乎可以无脑添加到backbone中。根据论文，SE block应该加在Inception block之后，ResNet网络应该加在shortcut之前，将前后对应的通道数对应上即可      

除了通道权重，CBAM还考虑空间权重，即：图像中心区域比周围区域更重要，由此设置不同位置的空间权重。CBAM将空间注意力和通道注意力结合起来。     
![alt text](assets_picture/conv_activate_token_loss/image-26.png)    
输入特征图F，经过两个并行的最大值池化和平均池化将C×H×W的特征图变成C×1×1的大小    
经过一个共享神经网络Shared MLP(Conv/Linear，ReLU，Conv/Linear)，压缩通道数C/r (reduction=16)，再扩张回C，得到两个激活后的结果。     
最后将二者相加再接一个sigmoid得到权重channel_out，再加权求和。      

此步骤与SENet不同之处是加了一个并行的最大值池化，提取到的高层特征更全面，更丰富。       

将上一步得到的结果通过最大值池化和平均池化分成两个大小为H×W×1的张量，然后通过Concat操作将二者堆叠在一起(C为2)，再通过卷积操作将通道变为1同时保证H和W不变，经过一个sigmoid得到spatial_out，最后spatial_out乘上一步的输入变回C×H×W，完成空间注意力操作

总结：

实验表明：通道注意力在空间注意力之前效果更好
加入CBAM模块不一定会给网络带来性能上的提升，受自身网络还有数据等其他因素影响，甚至会下降。如果网络模型的泛化能力已经很强，而你的数据集不是benchmarks而是自己采集的数据集的话，不建议加入CBAM模块。要根据自己的数据、网络等因素综合考量。      





### Depth wise和Pointwise降低运算量
？？？？？？？？      
卷积到底是一个卷积核乘以所有通道取平均？（或者说所有通道的单次卷积核一样参数？）还是有不一样的卷积核参数分别乘每个通道再相加？            
传统卷积：一个卷积核卷积图像的所有通道，参数过多，运算量大。     
![alt text](assets_picture/conv_activate_token_loss/image-23.png)     
![alt text](assets_picture/conv_activate_token_loss/1710589629695.png)     
Depth wise卷积：一个卷积核只卷积一个通道。输出图像通道数和输入时不变。缺点是每个通道独立卷积运算，没有利用同一位置上不同通道的信息       
Pointwise卷积：使用多个1×1标准卷积，将Depth wise卷积结果的各通道特征加权求和，得到新的特征图    
![alt text](assets_picture/conv_activate_token_loss/image-24.png)     
![alt text](assets_picture/conv_activate_token_loss/1710589645980.png)       

Group Conv组卷积      
下图假设卷积核大小为k×k，输入矩阵channel数为 c in
，卷积核个数为n（输出矩阵channel数）。组卷积分成g个组（group）      
![alt text](assets_picture/conv_activate_token_loss/image-27.png)      



### 为什么卷积核没有偶数的？     
看不懂       






## conv3d
3D conv的卷积核就是( c , k d , k h , k w )，其中k_d就是多出来的第三维，根据具体应用，在视频中就是时间维，在CT图像中就是层数维.   

举一个简单的例子，对于一个宽高均为[28,28]尺寸的彩色（3通道）视频数据集，假设每一次想要处理10帧图像，每次传入一份数据（10帧），那么输入尺寸为[1,3,10,28,28]。   

![Alt text](assets_picture/conv/image-31.png)   
![Alt text](assets_picture/conv/image-32.png)


$  out(N_i, C_{out_j}) = bias(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k) $   

where $\star$ is the valid 3D `cross-correlation`_ operator

torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)   

stride controls the stride for the cross-correlation.
stride 控制互相关的步幅。

padding controls the amount of padding applied to the input. It can be either a string {‘valid’, ‘same’} or a tuple of ints giving the amount of implicit padding applied on both sides.
padding 控制应用于输入的填充量。它可以是字符串 {‘valid’, ‘same’} 或整数元组，给出应用于两侧的隐式填充量。

dilation controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this link has a nice visualization of what dilation does.
dilation 控制内核点之间的间距；也称为 à trous 算法。描述起来比较困难，但这个链接很好地展示了 dilation 的作用。

groups controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups. For example,
groups 控制输入和输出之间的连接。 in_channels 和 out_channels 必须都能被 groups 整除。例如，

At groups=1, all inputs are convolved to all outputs.
当 groups=1 时，所有输入都与所有输出进行卷积。

At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels and producing half the output channels, and both subsequently concatenated.
在 groups=2 时，该操作等效于并排有两个卷积层，每个层看到一半的输入通道并产生一半的输出通道，并且随后将两者连接起来。

At groups= in_channels, each input channel is convolved with its own set of filters (of size 
out_channels/
in_channels
 ).
在 groups= in_channels 处，每个输入通道都与其自己的一组滤波器（大小为 
out_channels/
in_channels
  ）进行卷积。

The parameters kernel_size, stride, padding, dilation can either be:
参数 kernel_size 、 stride 、 padding 、 dilation 可以是：

a single int – in which case the same value is used for the depth, height and width dimension
单个 int – 在这种情况下，深度、高度和宽度尺寸使用相同的值

a tuple of three ints – in which case, the first int is used for the depth dimension, the second int for the height dimension and the third int for the width dimension
三个 int 的 tuple – 在这种情况下，第一个 int 用于深度尺寸，第二个 int 用于高度尺寸，第三个 int 用于宽度尺寸

depthwise convolution    
When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also known as a “depthwise convolution”.
当 groups == in_channels 且 out_channels == K * in_channels 时，其中 K 是正整数，此操作也称为“深度卷积”。   
![Alt text](assets_picture/conv/image-33.png)

## 池化
nn.MaxPool2d——最大池化：选取池化核覆盖区域的最大值作为输出。   
输出图像的尺寸计算公式：   
和卷积公式一样        
![Alt text](assets_picture/conv_activate_token_loss/image.png)   


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
 
对于深层网络，sigmoid函数反向传播时，很容易就会出现 梯度消失 的情况（在sigmoid接近**饱和区时，变换太缓慢，导数趋于0**，这种情况会造成信息丢失），从而无法完成深层网络的训练。


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

第一，**计算量**。采用sigmoid等函数，算激活函数时（指数运算），计算量大，反向传播求误差梯度时，求导涉及除法，计算量相对大，而采用Relu激活函数，整个过程的计算量节省很多。

第二，***梯度消失**。对于深层网络，sigmoid函数反向传播时，很容易就会出现 梯度消失 的情况（在sigmoid接近**饱和区时，变换太缓慢，导数趋于0**，这种情况会造成信息丢失），从而无法完成深层网络的训练。

第三，**稀疏性**。ReLu会使一部分神经元的输出为0，这样就造成了 网络的稀疏性，并且减少了参数的相互依存关系，缓解了**过拟合**问题的发生。

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

$$
\begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

$$

## softmax
![alt text](assets_picture/conv_activate_token_loss/image-20.png)      
softmax函数与max函数不同，max函数只输出最大值，而softmax确保较小的值具有较小的概率，并不会直接丢弃，可以认为它是argmax函数的概率版或soft版本，但这个soft其实并不那么soft。softmax函数的分母结合了输出值的所有因子，这意味着softmax函数获得的各种概率彼此相关，但softmax函数的缺点是在零点不可微，负输入的梯度为零，这意味着对于该区域的激活，权重不会在反向传播期间更新，因此会产生拥不激活的死亡神经元。        







## 交叉熵损失函数
![Alt text](assets_picture/conv/image-12.png)   
交叉熵损失函数    
交叉熵是用来衡量两个概率分布的距离。    
![alt text](assets_picture/detect_track_keypoint/image-11.png)    
交叉熵+sigmoid 输出为单值输出，可用于二分类。  
交叉熵+softmax 输出为多值，可用于多分类。   


## 均方误差
mse   
![alt text](assets_picture/conv_activate_token_loss/image-1.png)   
均方差是预测值与真实值之差的平方和，再除以样本量。
均方差函数常用于线性回归，即函数拟合。
用以回归任务：包括yolo回归坐标点，sd训练unet时对比预测和原始图像差异    
回归问题与分类问题不同，分类问题是判断一个物体在固定的n个类别中是哪一类。   
回归问题是对具体数值的预测。    
比如房价预测，销量预测等都是回归问题，这些问题需要预测的不是一个事先定义好的类别，而是一个任意实数。   

为什么使用交叉熵作分类，而不用作回归？   
可以替代均方差+sigmoid组合解决梯度消失问题，另外交叉熵只看重正确分类的结果，而均方差对每个输出结果都看重。？？？？？    
均方差为什么不适用于分类问题，而适用于回归？？？    
因为经过sigmoid函数后容易造成梯度消失，所以不适用于分类问题。    
均方差适用于线性的输出，特点是与真实结果差别越大，则惩罚力度越大。   

区别    
在一个三分类模型中，模型的输出结果为（a,b,c)，而真实的输出结果为(1,0,0)，那么MSE与cross-entropy相对应的损失函数的值如下：    
![alt text](assets_picture/conv_activate_token_loss/image-2.png)    
交叉熵的损失函数只和分类正确的预测结果有关系，而MSE的损失函数还和错误的分类有关系。     
均方差分类函数除了让正确的分类尽量变大，还会让错误的分类变得平均。    
实际在分类问题中这个调整是没有必要的，错误的分类不必处理。      
但是对于回归问题来说，这样的考虑就显得很重要了，回归的目标是输出预测值，而如果预测值有偏差，是一定要进行调整的。    
所以，回归问题使用交叉熵并不合适。    
![alt text](assets_picture/conv_activate_token_loss/image-3.png)    



## logistic回归
逻辑回归根据给定的自变量数据集来估计事件的发生概率，由于结果是一个概率，因此因变量的范围在 0 和 1 之间。    
![alt text](assets_picture/conv_activate_token_loss/image-4.png)     
广义的线性回归分析模型，常用于数据挖掘，疾病自动诊断，经济预测等领域    
logistic回归是一种广义线性回归（generalized linear model），因此与多重线性回归分析有很多相同之处。它们的模型形式基本上相同，都具有 w‘x+b，其中w和b是待求参数，其区别在于他们的因变量不同，多重线性回归直接将w‘x+b作为因变量，即y =w‘x+b，而logistic回归则通过函数L将w‘x+b对应一个隐状态p，p =L(w‘x+b),然后根据p 与1-p的大小决定因变量的值。如果L是logistic函数，就是logistic回归，如果L是多项式函数就是多项式回归。 [1]    
logistic回归的因变量可以是二分类的，也可以是多分类的，但是二分类的更为常用，也更加容易解释，多类可以使用softmax方法进行处理。实际中最为常用的就是二分类的logistic回归。 [1]     


## PNSR（Peak Signal-to-Noise Ratio峰值信噪比）
![Alt text](assets_picture/conv/image-13.png)    
![Alt text](assets_picture/conv/image-24.png)      
MAX表示像素值的最大可能取值（例如，对于8位图像，MAX为255），MSE是原始图像与重建图像之间的均方误差。   
是一种常用于衡量图像或视频质量的指标。它用于比较原始图像与经过处理或压缩后的图像之间的差异   
PSNR通过计算原始图像与重建图像之间的均方误差（Mean Squared Error，MSE）来量化它们之间的差异。    
PSNR的`值越高，表示图像的质量与原始图像的相似度越高`。常见的`PSNR范围通常在20到50之间`，数值越高表示图像质量越好。然而，PSNR作为一种图像质量评估指标也有其局限性。      
和直接用mse,不用log有什么区别？？？？         

它`主要关注均方误差`，忽略了人眼对于不同频率成分的敏感度差异以及感知失真的影响。因此，在某些情况下，PSNR可能不能准确地反映人类感知到的图像质量差异。   
了PSNR，还有其他更全面和准确的图像质量评估指标，例如结构相似性指标（Structural Similarity Index，SSIM）、感知质量评估指标（Perceptual Quality Assessment，如VIF、MSSSIM）等，这些指标综合考虑了人眼感知和图像结构信息，能够提供更全面的图像质量评估。   

意义：  
PSNR接近 50dB ，代表压缩后的图像仅有些许非常小的误差。  
PSNR大于 30dB ，人眼很难查觉压缩后和原始影像的差异。  
PSNR介于 20dB 到 30dB 之间，人眼就可以察觉出图像的差异。  
PSNR介于 10dB 到 20dB 之间，人眼还是可以用肉眼看出这个图像原始的结构，且直观上会判断两张图像不存在很大的差异。   
PSNR低于 10dB，人类很难用肉眼去判断两个图像是否为相同，一个图像是否为另一个图像的压缩结果。  


## MSSSIM（Multi-Scale Structural Similarity Index）
综合考虑了人眼感知和图像结构信息       
![Alt text](assets_picture/conv/image-14.png)    
是一种用于评估图像质量的指标，它是结构相似性指数（SSIM）在多个尺度上的扩展。   
SSIM是一种衡量两幅图像相似性的指标，它考虑了图像的**亮度、对比度和结构**等方面。而MS-SSIM在SSIM的基础上引入了多个尺度，以更好地捕捉图像的细节信息。    
具体而言，MS-SSIM的计算过程如下：

将原始图像和重建图像划分为不同尺度的子图像。

对每个尺度的子图像计算SSIM指数。

对每个尺度的SSIM指数进行加权平均，得到最终的MS-SSIM值。   

MS-SSIM的`值范围在0到1`之间，`数值越接近1表示重建图像与原始图像的相似度越高`，图像质量越好。

相比于PSNR，MS-SSIM考虑了图像的结构信息，能够更好地反映人眼对图像质量的感知。它在评估图像质量方面具有更高的准确性和敏感性。

需要注意的是，MS-SSIM计算复杂度相对较高，因为它需要对图像进行多尺度的分解和计算。然而，由于其良好的性能，在图像压缩、图像处理等领域得到广泛应用，并且被认为是一种较为可靠的图像质量评估指标。  

```python
import cv2
import numpy as np
 
def ms_ssim(img1, img2):
    # 转换为灰度图像
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 计算MS-SSIM
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])  # 不同尺度的权重
    levels = weights.size
    
    mssim = np.zeros(levels)
    mcs = np.zeros(levels)
    
    for i in range(levels):
        ssim_map, cs_map = ssim(img1, img2)
        mssim[i] = np.mean(ssim_map)
        mcs[i] = np.mean(cs_map)
        
        img1 = cv2.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (img2.shape[1] // 2, img2.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    
    # 整体MS-SSIM计算
    overall_mssim = np.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1])
    #"np.prod"是一个函数，用于计算给定数组中所有元素的乘积 
    #arr = np.array([1, 2, 3, 4, 5])
    #product = np.prod(arr)
    #print(product)  # 输出结果为120，即1*2*3*4*5
    return overall_mssim
 
def ssim(img1, img2, k1=0.01, k2=0.03, win_size=11, L=255):
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    
    # 计算均值和方差
    mu1 = cv2.GaussianBlur(img1, (win_size, win_size), 1.5)
    #对图像进行高斯模糊处理。具体来说，它将图像img1应用高斯模糊，并指定了模糊核的大小(win_size, win_size)和高斯核的标准差(sigma)。这个函数会返回一个模糊后的图像
    #？？？
    #(11, 11)：指定了高斯核的大小为11x11，这个值决定了模糊程度。1.5：是高斯核的标准差，用来控制模糊程度。标准差越大，模糊程度越高。
    mu2 = cv2.GaussianBlur(img2, (win_size, win_size), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (win_size, win_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (win_size, win_size), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (win_size, win_size), 1.5) - mu1_mu2
    
    # 计算相似性度量
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    
    return ssim_map, cs_map
 
# 读取图像
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
 
# 计算MS-SSIM
ms_ssim_score = ms_ssim(img1, img2)
print("MS-SSIM score:", ms_ssim_score)
```

评价加权分数一般采用  
![Alt text](assets_picture/conv/image-15.png)   

## LPIPS (Learned Perceptual Image Patch Similarity）
具体怎么计算？？？         
计算特征图的mse??????        

可学习感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS)也称为“感知损失”(perceptual loss)   
是一种基于学习的感知图像补丁相似性指标，用于评估图像的感知质量。   

具体而言，LPIPS的计算过程如下：   

使用预训练的CNN模型（通常是基于深度学习的图像分类模型）提取原始图像和重建图像的特征表示。

将提取的特征表示作为输入，通过一个距离度量函数计算图像之间的相似性得分。

相似性得分表示图像之间在感知上的差异，数值越小表示图像之间的感知差异越小，图像质量越好。

LPIPS的得分范围通常是0到1之间，数值越小表示图像的感知质量越高。  
与传统的图像质量评估指标（如PSNR和SSIM）相比，LPIPS更加注重于人眼感知的因素，能够更好地捕捉到图像之间的感知差异。它在图像生成、图像编辑等任务中被广泛应用，特别适用于需要考虑感知质量的场景。  
需要注意的是，LPIPS是一种基于学习的指标，它的性能受到所使用的CNN模型和训练数据的影响。因此，在使用LPIPS进行图像质量评估时，**需要使用与训练模型相似的数据集和预训练模型**，以保证评估结果的准确性和可靠性。    

LPIPS的设计灵感来自于人眼对图像的感知，它通过学习一个神经网络模型来近似人类感知的视觉相似性。该模型使用卷积神经网络（CNN）对图像的局部补丁进行特征提取，并计算补丁之间的相似性得分。   
LPIPS 比传统方法（比如L2/PSNR, SSIM, FSIM）更符合人类的感知情况。LPIPS的值越低表示两张图像越相似，反之，则差异越大。    
![Alt text](assets_picture/conv/image-25.png)   
将左右的两个图像块和中间的图像块进行比较：  
如图表示，每一组有三张图片，由传统的评价标准如L2、SSIM、PSNR等评价结果和人体认为的大不相同，这是传统方法的弊端。如果图片平滑，那么传统的评价方式则大概率会失效。      
而目前GAN尤其是VAE等生成模型生成结果都过于平滑。 而最后三行的评价为深度学习的方式，可以看到，通过神经网络（非监督、自监督、监督模型）提取特征的方式，并对特征差异进行计算能够有效进行评价，而且能够和人体评价相似。   





PSNR主要用于衡量图像的重建误差，而MS-SSIM和LPIPS更加关注人眼对图像感知的差异。在实际应用中，不同的指标可以结合使用，以综合评估图像质量。  

## rFID  
(Fréchet Inception Distance)   
计算Frechet distance between 2 Gaussians (训练好的图片分类的模型的CNN去除 真实和生成的respresentations，计算距离)  
需要大量样本一次性计算  
![Alt text](assets_picture/stable_diffusion/image-49.png)  
- 从真实图像和生成图像中分别抽取n个随机子样本，并通过Inception-v3网络获得它们的特征向量。
- 计算真实图像子样本的特征向量的平均值mu1和协方差矩阵sigma1，以及生成图像子样本的特征向量的平均值mu2和协方差矩阵sigma2。
- 计算mu1和mu2之间的欧几里德距离d^2，以及sigma1和sigma2的平方根的Frobenius范数||sigma1^(1/2)*sigma2^(1/2)||_F。  
  - ![Alt text](assets_picture/stable_diffusion/image-117.png)   
  ![Alt text](assets_picture/stable_diffusion/image-118.png)   
  - 欧几里德距离 d = sqrt((x1-x2)^+(y1-y2)^)
  - Frobenius norm（弗罗贝尼乌斯-范数）（F-范数）  
  ![Alt text](assets_picture/stable_diffusion/image-51.png)  
  ![Alt text](assets_picture/stable_diffusion/image-50.png)    
  这个范数是针对矩阵而言的，具体定义可以类比 向量的L2范数
- 计算FID距离：FID = d^2 + ||sigma1^(1/2)*sigma2^(1/2)||_F。  


## 准确率(Precision)、召回率(Recall)、F值(F-Measure)、ROC曲线、PR曲线 
![alt text](assets_picture/conv_activate_token_loss/image-5.png)    
![alt text](assets_picture/conv_activate_token_loss/image-8.png)    

召回率（recall）    
所有实际正例中有多少被预测为正例     
对应漏检    
R= TP / (TP+FN)    

精确率、精度（Precision）   
表示被分为正例的示例中实际为正例的比例。    
对应误检     
精确率(precision)定义为：   
![alt text](assets_picture/conv_activate_token_loss/image-7.png)    

准确率（Accuracy）    
被分对的样本数除以所有的样本数     
![alt text](assets_picture/conv_activate_token_loss/image-6.png)    

综合评价指标（F-Measure）     
P和R指标有时候会出现的矛盾的情况，这样就需要综合考虑他们，最常见的方法就是F-Measure（又称为F-Score）。
F-Measure是Precision和Recall加权调和平均：    
![alt text](assets_picture/conv_activate_token_loss/image-9.png)     
当参数α=1时，就是最常见的F1，也即    
![alt text](assets_picture/conv_activate_token_loss/image-10.png)     

ROC曲线：    
ROC（Receiver Operating Characteristic）曲线是以假正率（FP_rate）和真正率（TP_rate）为轴的曲线，ROC曲线下面的面积我们叫做AUC   
![alt text](assets_picture/conv_activate_token_loss/image-11.png)   

PR曲线：   
假设N_c>>P_c（即Negative的数量远远大于Positive的数量），若FP很大，即有很多N的sample被预测为P，因为FP_{rate}=FP/N_c ，因此FP_rate的值仍然很小（如果利用ROC曲线则会判断其性能很好，但是实际上其性能并不好），但是如果利用PR，因为Precision综合考虑了TP和FP的值，因此在极度不平衡的数据下（Positive的样本较少），PR曲线可能比ROC曲线更实用。
![alt text](assets_picture/conv_activate_token_loss/image-12.png)         
简单来说，AP就是PR曲线与坐标轴围成的面积，可表示为，    
![alt text](assets_picture/conv_activate_token_loss/image-13.png)         
实际计算中，我们并不直接对该PR曲线进行计算，而是对PR曲线进行平滑处理。即对PR曲线上的每个点，Precision的值取该点右侧最大的Precision的值，如下图所示。      
![alt text](assets_picture/conv_activate_token_loss/image-14.png)     

锚框Anchor：最初的YOLO模型相对简单，没有采用锚点，而最先进的模型则依赖于带有锚点的两阶段检测器。YOLOv2采用了锚点，从而提高了边界盒的预测精度。这种趋势持续了五年，直到YOLOX引入了一个无锚的方法，取得了最先进的结果。从那时起，随后的YOLO版本已经放弃了锚的使用；    
骨干网络Backbone：YOLO模型的骨干架构随着时间的推移发生了重大变化。从由简单的卷积层和最大集合层组成的Darknet架构开始，后来的模型在YOLOv4中加入了跨阶段部分连接（CSP），在YOLOv6和YOLOv7中加入了重新参数化，并在DAMO-YOLO中加入了神经架构搜索；     

![alt text](assets_picture/conv_activate_token_loss/image-15.png)       
![alt text](assets_picture/conv_activate_token_loss/image-16.png)        








## NAFNet -- Nonlinear Activation Free Network
![Alt text](assets_picture/conv/image-16.png)   
网络主干继承于NAFNet，为了减少推理时间，做了如下调整：

1. 网络结构变动： 

 * 网络第一层只采用一次 3x3 卷积提高通道
 
 * 每层做通道放缩，网络第一层通过3x3卷积将通道由3扩大至width=12，最后一层再缩小至原始图片通道数3， 模型大小3.1m  
 在编码器的3个阶段使用性能更好的LKABlock，设置为[1, 3, 4, 0]，而在解码器阶段使用相对简单的NAFBlock，设置为[1, 1, 1, 0]。   
![Alt text](assets_picture/conv/image-17.png)

2. 引入 LKA - 大核卷积注意力机制  
LKA大核注意力模块包含大感受野和1x1点卷积提取局部信息的优势
由于通道数的下降和middle_blk的取消， 模型的视野阈会受到影响， 选择 VAN 网络的 LKA 结构来缓解影响，验证可得选择LKA结构，模型收敛速度较原始nafnet网络的收敛速度能进一步提高。  
解码器是attention + mlp结构

3. 使用 Layernorm2D 而非 BatchNorm，对特征图每一层计算均值和方差   

## LKA
前有微软 SwinTransformer引入CNN的滑动窗口等特性，刷榜下游任务并获马尔奖。   
后有Meta AI的 ConvNeXT 用ViT上的大量技巧魔改ResNet后实现性能反超 。   
现在一种全新Backbone—— VAN（Visiual Attention Network, 视觉注意力网络）再次引起学界关注。     
因为新模型再一次 刷榜三大视觉任务，把上面那两位又都给比下去了。  

VAN号称同时吸收了CNN和ViT的优势且简单高效，精度更高的同时参数量和计算量还更小。
大内核注意力（LKA）模块的视觉注意力网络（VAN）已被证明在一系列基于视觉的任务中具有超越视觉转换器（ViT）的卓越性能。然而，随着卷积核大小的增加，这些 LKA 模块中的深度卷积层会导致计算和内存占用的二次方增加。为了缓解这些问题，并在 VAN 的注意力模块中使用超大卷积核，我们提出了大型可分离核注意力模块系列，称为 LSKA。  
![Alt text](assets_picture/conv/image-18.png)    
![Alt text](assets_picture/conv/image-19.png)  
在图像分类、物体检测和语义分割方面，带有 LKA 的 VAN 已被证明优于最先进的 ViT 和 CNN。然而，大规模深度卷积核的设计仍然会产生高计算量和内存占用，随着核大小的增加，模型的有效性也会降低。当核大小达到 35 × 35 和 53 × 53 时，在 VAN 的 LKA 模块中设计深度卷积（不使用深度扩张卷积）的计算效率很低。  
为了提高计算效率，LKA 采用了带深度卷积（DW-D-Conv）的扩张卷积，以获得更大的 ERFs。  
在图像分类、物体检测和语义分割方面，VAN 甚至比 PVT-V2、Swin Transformer 和 Twins-SVT 等一系列变压器网络取得了更好的性能。   
注意机制用于选择图像中最重要的区域。一般来说，可分为四类：空间注意 ；通道注意；时间注意 和分支注意 。在此，我们更关注通道注意和空间注意，因为它们与我们的工作更为相关。通道注意力侧重于特定模型层的 "什么 "语义属性。由于特征图的每个通道都是一个检测器（也称为滤波器）的响应图，通道注意力机制允许模型将注意力集中在各通道中物体的特定属性上。与通道注意力不同，空间注意力关注的是模型应该关注的语义相关区域的 “位置”。   

## dw-conv 深度卷积 (Depthwise Convolution)
如何去捕捉长距离的依赖呢？
有两种常见的方法：1) 使用自注意力机制。在研究动机中已经讲述了在视觉中使用自注意力机制的不足。2) 使用大核卷积来捕捉长距离依赖。使用该方法的不足在于，大卷积核的参数量和计算量太大，难以接受。

本文针对 2) 进行了改进，提出了一种新的分解方式，用于减少大卷积的计算量和参数量。

如图2所示：我们可以将一个Kx K 的大卷积分解成三部分: 

a) 一个(K/d) x (K/d) 的depth-wise dilation convolution，其中dilation的大小为d;

b) 一个(2d-1)x (2d-1) 的 depth-wise convolution；

c) 一个 1x1 卷积。

这种分解可以理解为如何选择三种基本的构件来布满整个卷积空间。图2展示了将一个 13 x 13 的卷积分解成一个 5 x 5 的 depth-wise convolution 、一个 5 x 5 的depth-wise dilation convolution，和一个 1 x 1 的卷积，其中 d = 3 。

![Alt text](assets_picture/conv/image-20.png)

具体来说，如果使用一个大小为 k 的卷积核，其参数数量为 k * k * c_in * c_out，其中 c_in 表示输入的通道数，c_out 表示输出的通道数。而使用 DWConv，其参数数量则为 k * k * c_in，因为不同通道共享相同的卷积核。

![Alt text](assets_picture/conv/image-21.png)    
Depthwise Convolution is a type of convolution where we apply a single convolutional filter for each input channel. In the regular 2D convolution performed over multiple input channels, the filter is as deep as the input and lets us freely mix channels to generate each element in the output. In contrast, depthwise convolutions keep each channel separate. To summarize the steps, we:    
深度卷积是一种卷积，我们为每个输入通道应用单个卷积滤波器。在多个输入通道上执行的常规 2D 卷积中，滤波器的深度与输入一样深，让我们可以自由混合通道以生成输出中的每个元素。相反，深度卷积使每个通道保持分离。总结一下这些步骤，我们：    
Split the input and filter into channels.   
将输入和滤波器拆分为通道。   
We convolve each input with the respective filter.   
我们将每个输入与相应的过滤器进行卷积。   
We stack the convolved outputs together.
我们将卷积输出堆叠在一起。   

而深度卷积每个卷积核都是单通道的，维度为(1,1,k,k) ，卷积核的个数为iC（须和feature map的通道数保持一致），即第i个卷积核与feature map第i个通道进行二维的卷积计算，最后输出维度为(1,iC,oH,oW)   
所以通常会在深度卷积后面接上一个(oC,iC,1,1)的标准卷积来代替3×3或更大尺寸的标准卷积。总的计算量为k×k×iC×oH×oW+iC×1×1×oH×oW×oC，是普通卷积的1/oC+1/(k×k)，大大减少了计算量和参数量，又可以达到相同的效果，这种结构被称为深度可分离卷积(Depthwise Separable Convolution)，在MobileNet V1被提出，后来渐渐成为轻量化结构设计的标配。
![Alt text](assets_picture/conv/image-23.png)   
1×1卷积（Pointwise Convolution）  


## 可变形卷积 (Deformable Convolution)
以上的卷积计算都是固定的，每次输入不同的图像数据，卷积计算的位置都是完全固定不变，即使是空洞卷积/转置卷积，0填充的位置也都是事先确定的。而可变形卷积是指卷积核上对每一个元素额外增加了一个h和w方向上偏移的参数，然后根据这个偏移在feature map上动态取点来进行卷积计算，这样卷积核就能在训练过程中扩展到很大的范围。而显而易见的是可变性卷积虽然比其他卷积方式更加灵活，可以根据每张输入图片感知不同位置的信息，类似于注意力，从而达到更好的效果。   
![Alt text](assets_picture/conv/image-22.png)     



## 分词器  tokenizer 
由于神经网络模型不能直接处理文本，因此我们需要先将文本转换为数字，这个过程被称为编码 (Encoding)，其包含两个步骤：

使用分词器 (tokenizer) 将文本按词、子词、字符切分为 tokens；   
将所有的 token 映射到对应的 token ID。   

子词分词法有很多不同取得最小可分子词的方法，例如BPE（Byte-Pair Encoding，字节对编码法），WordPiece，SentencePiece，Unigram等等    

分词策略   
根据切分粒度的不同，分词策略可以分为以下几种：



### 按词切分 (Word-based)   
![Alt text](assets_picture/conv/image-26.png)   
这种策略的问题是会将文本中所有出现过的独立片段都作为不同的 token，从而产生巨大的词表。而实际上很多词是相关的，例如 “dog” 和 “dogs”、“run” 和 “running”，如果给它们赋予不同的编号就无法表示出这种关联性。

词表就是一个映射字典，负责将 token 映射到对应的 ID（从 0 开始）。神经网络模型就是通过这些 token ID 来区分每一个 token。

当遇到不在词表中的词时，分词器会使用一个专门的 unk
 token 来表示它是 unknown 的。显然，如果分词结果中包含很多 unk
 就意味着丢失了很多文本信息，因此一个好的分词策略，应该尽可能不出现 unknown token。

 ### 按字符切分 (Character-based)   
 ![Alt text](assets_picture/conv/image-27.png)    
 这种策略把文本切分为字符而不是词语，这样就只会产生一个非常小的词表，并且很少会出现词表外的 tokens。

但是从直觉上来看，字符本身并没有太大的意义，因此将文本切分为字符之后就会变得不容易理解。这也与语言有关，例如中文字符会比拉丁字符包含更多的信息，相对影响较小。此外，这种方式切分出的 tokens 会很多，例如一个由 10 个字符组成的单词就会输出 10 个 tokens，而实际上它们只是一个词。

因此现在广泛采用的是一种同时结合了按词切分和按字符切分的方式——按子词切分 (Subword tokenization)。    

### **按子词切分 (Subword) **
子词分词法有很多不同取得最小可分子词的方法，例如BPE（Byte-Pair Encoding，字节对编码法），WordPiece，SentencePiece，Unigram等等   
多语言支持：Sentence-Piece    
Sentence-Piece，其实是HF里面大量模型会调用的包，例如ALBERT，XLM-RoBERTa和T5：   
这个包主要是为了多语言模型设计的，它做了两个重要的转化：    
以unicode方式编码字符，将所有的输入（英文、中文等不同语言）都转化为unicode字符，解决了多语言编码方式不同的问题。  
将空格编码为‘_’， 如'New York' 会转化为['_', 'New', '_York']，这也是为了能够处理多语言问题，比如英文解码时有空格，而中文没有， 这种语言区别。  


力求trade off，存储最少，运算最少，意义最大，unk最少    

高频词直接保留，低频词被切分为更有意义的子词。例如 “annoyingly” 是一个低频词，可以切分为 “annoying” 和 “ly”，这两个子词不仅出现频率更高，而且词义也得以保留。下图展示了对 “Let’s do tokenization!“ 按子词切分的结果：

![Alt text](assets_picture/conv/image-28.png)

可以看到，“tokenization” 被切分为了 “token” 和 “ization”，不仅保留了语义，而且只用两个 token 就表示了一个长词。这种策略只用一个较小的词表就可以覆盖绝大部分文本，基本不会产生 unknown token。尤其对于土耳其语等黏着语，几乎所有的复杂长词都可以通过串联多个子词构成。    

  


调用 Tokenizer.save_pretrained() 函数会在保存路径下创建三个文件：

special_tokens_map.json：映射文件，里面包含 unknown token 等特殊字符的映射关系；   
tokenizer_config.json：分词器配置文件，存储构建分词器需要的参数；   
vocab.txt：词表，一行一个 token，行号就是对应的 token ID（从 0 开始）。   

### BERT 分词器  Word-Piece 
BERT族：Word-Piece   
Word-Piece和BPE非常相似   
BERT在使用Word-Piece时加入了一些特殊的token，例如[CLS]和[SEP]   

```
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)

['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']


ids = tokenizer.convert_tokens_to_ids(tokens)

[7993, 170, 13809, 23763, 2443, 1110, 3014]

```
可以看到，BERT 分词器采用的是子词切分策略，它会不断切分词语直到获得词表中的 token，例如 “transformer” 会被切分为 “transform” 和 “##er”。    

前面说过，文本编码 (Encoding) 过程包含两个步骤：

分词：使用分词器按某种策略将文本切分为 tokens；   
映射：将 tokens 转化为对应的 token IDs。

```
sequence_ids = tokenizer.encode(sequence)

包括但不限于，同时将cls和sep自动添加到首尾
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)


实际使用，直接
tokenized_text = tokenizer("Using a Transformer network is simple")

这样不仅会返回分词后的 token IDs，还包含模型需要的其他输入。例如 BERT 分词器还会自动在输入中添加 token_type_ids 和 attention_mask：   
{'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```


文本解码 (Decoding) 与编码相反，负责将 token IDs 转换回原来的字符串。注意，解码过程不是简单地将 token IDs 映射回 tokens，还需要合并那些被分为多个 token 的单词。


### Padding 操作
按批输入多段文本产生的一个直接问题就是：batch 中的文本有长有短，而输入张量必须是严格的二维矩形，维度为 [bs,seq len]
，即每一段文本编码后的 token IDs 数量必须一样多。例如下面的 ID 列表是无法转换为张量的：
```
batched_ids = [
    [200, 200, 200],
    [200, 200]
]
```
我们需要通过 Padding 操作，在短序列的结尾填充特殊的 padding token   

### Attention Mask
```
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward0>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
tensor([[ 1.5694, -1.3895],
        [ 1.3374, -1.2163]], grad_fn=<AddmmBackward0>)
```
使用 padding token 填充的序列的结果与其单独送入模型时不同   
模型默认会编码输入序列中的所有 token 以建模完整的上下文，因此这里会将填充的 padding token 也一同编码进去，从而生成不同的语义表示。   
```
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
batched_attention_masks = [
    [1, 1, 1],
    [1, 1, 0],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
outputs = model(
    torch.tensor(batched_ids), 
    attention_mask=torch.tensor(batched_attention_masks))
print(outputs.logits)
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward0>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
```
在实际使用时，我们应该直接使用分词器对文本进行处理，它不仅会向 token 序列中添加模型需要的特殊字符（例如 cls,sep
），还会自动生成对应的 Attention Mask。

目前大部分 Transformer 模型只能接受长度不超过 512 或 1024 的 token 序列，因此对于长序列，有以下三种处理方法：

使用一个支持长文的 Transformer 模型，例如 Longformer 和 LED（最大长度 4096）；   
设定最大长度 max_sequence_length 以截断输入序列：sequence = sequence[:max_sequence_length]。   
将长文切片为短文本块 (chunk)，然后分别对每一个 chunk 编码。在后面的快速分词器中，我们会详细介绍。   

### 编码句子对   
此时分词器会使用 se[]
 token 拼接两个句子，输出形式为“cls 1 sep 2 sep
”的 token 序列，这也是 BERT 模型预期的“句子对”输入格式。
```
inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
print(tokens)
{'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
```
如果我们选择其他模型，分词器的输出不一定会包含 token_type_ids 项（例如 DistilBERT 模型）。分词器只需保证输出格式与模型预训练时的输入一致即可。

句子对例子，三条，每条两句
```
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentence1_list = ["First sentence.", "This is the second sentence.", "Third one."]
sentence2_list = ["First sentence is short.", "The second sentence is very very very long.", "ok."]

tokens = tokenizer(
    sentence1_list,
    sentence2_list,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
print(tokens)
print(tokens['input_ids'].shape)


{'input_ids': tensor([[ 101, 2034, 6251, 1012,  102, 2034, 6251, 2003, 2460, 1012,  102,    0,
            0,    0,    0,    0,    0,    0],
        [ 101, 2023, 2003, 1996, 2117, 6251, 1012,  102, 1996, 2117, 6251, 2003,
         2200, 2200, 2200, 2146, 1012,  102],
        [ 101, 2353, 2028, 1012,  102, 7929, 1012,  102,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
torch.Size([3, 18])

```


句子对例子，三条，每条三句   
没有意义，被当成是标签  
```
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentence1_list = ["First sentence.", "This is the second sentence.", "Third one."]
sentence2_list = ["First sentence is short.", "The second sentence is very very very long.", "ok."]
sentence3_list = ["First sentence is short.", "The second sentence is very very very long.", "ok."]
tokens = tokenizer(
    sentence1_list,
    sentence2_list,
    sentence3_list,  被误认为是标签
    padding=True,
    truncation=True,
    return_tensors="pt"
)
print(tokens)
print(tokens['input_ids'].shape)



{'input_ids': tensor([[ 101, 2034, 6251, 1012,  102, 2034, 6251, 2003, 2460, 1012,  102,    0,
            0,    0,    0,    0,    0,    0],
        [ 101, 2023, 2003, 1996, 2117, 6251, 1012,  102, 1996, 2117, 6251, 2003,
         2200, 2200, 2200, 2146, 1012,  102],
        [ 101, 2353, 2028, 1012,  102, 7929, 1012,  102,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'labels': tensor([[ 101, 2034, 6251, 2003, 2460, 1012,  102,    0,    0,    0,    0],
        [ 101, 1996, 2117, 6251, 2003, 2200, 2200, 2200, 2146, 1012,  102],
        [ 101, 7929, 1012,  102,    0,    0,    0,    0,    0,    0,    0]])}被误认为是标签
torch.Size([3, 18])

```

三个句子  
token_type_ids不做区分
```
inputs = tokenizer("This is the first sentence. [SEP] This is the second one. [SEP] This is the third one.")
print(inputs)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
print(tokens)


{'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102, 2023, 2003, 1996, 2353, 2028, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]', 'this', 'is', 'the', 'third', 'one', '.', '[SEP]']


```

### 添加新 token     
一些领域的专业词汇，例如使用多个词语的缩写拼接而成的医学术语，同样也不在模型的词表中，因此也会出现上面的问题。此时我们就需要将这些新 token 添加到模型的词表中    
向词表中添加新 token 后，必须重置模型 embedding 矩阵的大小，也就是向矩阵中添加新 token 对应的 embedding，这样模型才可以正常工作，将 token 映射到对应的 embedding。  
model.resize_token_embeddings(len(tokenizer))   
```
vocabulary size: 30522
After we add 2 tokens
vocabulary size: 30524
torch.Size([30524, 768])
```
在默认情况下，新添加 token 的 embedding 是随机初始化的。   
Token embedding 初始化   
如果有充分的语料对模型进行微调或者继续预训练，那么将新添加 token 初始化为随机向量没什么问题。但是如果训练语料较少，甚至是只有很少语料的 few-shot learning 场景下，这种做法就存在问题。研究表明，在训练数据不够多的情况下，这些新添加 token 的 embedding 只会在初始值附近小幅波动。换句话说，即使经过训练，它们的值事实上还是随机的。?????   
直接赋值   
初始化为已有 token 的值   

tokenize完的下一步就是将token的one-hot编码转换成更dense的embedding编码。  
在ELMo之前的模型中，embedding模型很多是单独训练的，而ELMo之后则爆发了直接将embedding层和上面的语言模型层共同训练的浪潮（ELMo的全名就是Embeddings from Language Model）。   


 



### GPT clip 分词器 BPE
GPT族：Byte-Pair Encoding (BPE)    
bpe     
   
BPE最初是用于文本压缩的算法，当前是最常见tokenizer的编码方法，用于 GPT (OpenAI) 和 Bert (Google) 的 Pre-training Model。  

```
1. 统计输入中所有出现的单词并在每个单词后加一个单词结束符</w> -> ['hello</w>': 6, 'world</w>': 8, 'peace</w>': 2]
2. 将所有单词拆成单字 -> {'h': 6, 'e': 10, 'l': 20, 'o': 14, 'w': 8, 'r': 8, 'd': 8, 'p': 2, 'a': 2, 'c': 2, '</w>': 3}
3. 合并最频繁出现的单字(l, o) -> {'h': 6, 'e': 10, 'lo': 14, 'l': 6, 'w': 8, 'r': 8, 'd': 8, 'p': 2, 'a': 2, 'c': 2, '</w>': 3}
4. 合并最频繁出现的单字(lo, e) -> {'h': 6, 'lo': 4, 'loe': 10, 'l': 6, 'w': 8, 'r': 8, 'd': 8, 'p': 2, 'a': 2, 'c': 2, '</w>': 3}
5. 反复迭代直到满足停止条件
显然，这是一种贪婪的算法。在上面的例子中，'loe'这样的子词貌似不会经常出现，但是当语料库很大的时候，诸如est，ist，sion，tion这样的特征会很清晰地显示出来。

在获得子词词表后，就可以将句子分割成子词了，算法见下面的例子（引自文章）：

# 给定单词序列
["the</w>", "highest</w>", "mountain</w>"]

# 从一个很大的corpus中排好序的subword表如下
# 长度 6         5           4        4         4       4          2
["errrr</w>", "tain</w>", "moun", "est</w>", "high", "the</w>", "a</w>"]

# 迭代结果
"the</w>" -> ["the</w>"]
"highest</w>" -> ["high", "est</w>"]
"mountain</w>" -> ["moun", "tain</w>"]

```



#### bpe和wordpiece区别？？？
用于将词汇分割成更小的单元，从而能够更灵活地处理未登录词（out-of-vocabulary words）和稀有词汇   
（2）BPE与Wordpiece都是首先初始化一个小词表，再根据一定准则将不同的子词合并。词表由小变大*  
（3）BPE与Wordpiece的最大区别在于，如何选择两个子词进行合并：BPE选择频数最高的相邻子词合并，而WordPiece选择能够提升语言模型概率最大的相邻子词加入词表。  
（4）其实还有一个Subword算法，ULM（Unigram Language Model），与前两种方法不同的是，该方法先初始化一个大词表，再根据评估准则不断丢弃词表，直到满足限定条件为止，词表由大变小。  

##### BPE Byte-Pair Encoding
全称为字节对编码，是一种数据压缩方法，通过迭代地合并最频繁出现的字符或字符序列来实现分词目的。

算法步骤：

1. 准备足够大的训练语料
2. 确定期望的subword词表大小
3. 将单词拆分为字符序列并在末尾添加后缀“ </ w>”，统计单词频率。本阶段的subword的粒度是字符。例如单词“ low”的频率为5，那么我们将其改写为“ l o w </ w>”：5
4. 统计每一个连续字节对的出现频率，选择最高频者合并成新的subword
5. 重复第4步直到达到第2步设定的subword词表大小或下一个最高频的字节对出现频率为1   

停止符"</w>"的意义在于表示subword是词后缀。举例来说："st"字词不加"</w>"可以出现在词首如"st ar"，加了"</w>"表明改字词位于词尾，如"wide st</w>"，二者意义截然不同。    

首先让我们看看单个单词出现的频率，单词出现频率统计如下：   
![Alt text](assets_picture/conv/image-29.png)   
根据上图开始第一次迭代，由上图可以看出最频繁的字符对是“ d ”和“ e ”，共有3+2+1+1=7次。将字符对“ d ”和“ e ”组合起来创建第一个子词标记（不是单个字符）“ de ”。符合上述合并后词表可能出现3种变化中的第一条，增加合并后的新字词“ de ”，同时原来的2个子词还保留“ d ”和“ e ”也会保留下来(由于build和the两个单词)。在第一次迭代完成后的字符对词频更新为如下：   
![Alt text](assets_picture/conv/image-30.png)    
不断迭代，最终达到步骤二中期望的subword词表大小。

举个例子，我们假设在预标记化之后，已经确定了以下单词集（包括它们的频率）：

Copied
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)   

因此，基本词汇是 ["b", "g", "h", "n", "p", "s", "u"] 。将所有单词拆分为基本词汇表的符号，我们得到：  
 Splitting all words into symbols of the base vocabulary  
Copied
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)

然后，BPE 计算每个可能的符号对symbol pair 的频率，并选择出现最频繁的符号对。在上面的示例中， "h" 后跟 "u" 出现了 10 + 5 = 15 次（ "hug" 出现 10 次，出现 10 次； "hugs" 出现 5 次，出现 5 次）。然而，最常见的符号对是 "u"  "g" ，总共出现 10 + 5 + 5 = 20 次。因此，分词器学习的第一个合并规则是将所有 "u" 符号后跟 "g" 符号分组在一起。接下来， "ug" 被添加到词汇表中。那么单词集就变成了

Copied
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)

然后，BPE 识别下一个最常见的符号对symbol pair。 "u" 后面跟着 "n" ，出现了 16 次。 "u" 、 "n" 合并到 "un" 并添加到词汇表中。下一个最常见的符号对是 "h"  "ug" ，出现了 15 次。再次合并该对，并且可以将 "hug" 添加到词汇表中。

在这个阶段，词汇表是 ["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"] ，我们的独特单词集表示为

Copied
("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)

假设字节对编码训练将在此时停止，则学习的合并规则将应用于新单词（只要这些新单词不包含不在基本词汇表中的符号）。例如，单词 "bug" 将被标记为 ["b", "ug"] ，但 "mug" 将被标记为 ["unk>", "ug"] ，因为符号 "m" 不在基本词汇表中。一般来说，诸如 "m" 之类的单个字母不会被 "unk>" 符号替换，因为训练数据通常包含每个字母至少出现一次的情况，但对于表情符号等非常特殊的字符来说，这种情况很可能发生。

如前所述，词汇量大小，即基本词汇量+合并次数，是一个需要选择的超参数。例如，GPT 的词汇量为 40,478，因为它们有 478 个基本字符，并在 40,000 次合并后选择停止训练。

包含所有可能的基本字符base characters的基本词汇表base vocabulary可能会非常大，例如：所有 unicode 字符unicode characters都被视为基本字符。为了拥有更好的基础词汇base vocabulary，GPT-2 使用字节bytes作为基础词汇，这是一个巧妙的技巧，强制基础词汇base vocabulary的大小为 256，同时确保每个基础字符都包含在词汇中。通过一些处理标点符号的附加规则，GPT2 的分词器可以对每个文本进行分词，而不需要 unk> symbol符号。 GPT-2 的词汇量为 50,257，对应于 256 字节的基本标记256 bytes base tokens、特殊的文本结束标记以及通过 50,000 次合并学习的符号。




##### WordPiece
WordPiece算法可以看作是BPE的变种。不同点在于，WordPiece基于概率生成新的subword而不是下一最高频字节对。

与 BPE 非常相似。 WordPiece 首先初始化词汇表the vocabulary以包含训练数据中存在的每个字符character ，并逐步学习给定数量的合并规则。与 BPE 不同，WordPiece 不会选择最常见的符号对most frequent symbol pair，而是选择在添加到词汇表后使训练数据的可能性最大化的符号对symbol pair, that maximizes the likelihood of the training data once added to the vocabulary。

参考前面的例子，最大化训练数据的似然相当于​​找到一个符号对，其概率除以其第一个符号随后其第二个符号的概率是所有符号对symbol pairs中最大的。例如。仅当 "ug" 除以 "u" 、 "g" 的概率大于任何其他符号对时， "u" 和后跟 "g" 才会被合并。直观上，WordPiece 与 BPE 略有不同，它通过合并两个符号来评估其损失，以确保其值得。    
直观上，WordPiece 与 BPE 略有不同，它通过合并两个符号by merging two symbols来评估其损失，以确保其值得。？？？？？？？

算法步骤

1. 准备足够大的训练语料

2. 确定期望的subword词表大小

3. 将单词拆分成字符序列

4. 基于第3步数据训练语言模型？？？？？？？？    
那岂不是要训练很多次？？？？？具体要迭代多少次？？？？？       

5. 从所有可能的subword单元中选择加入语言模型后能最大程度地增加训练数据概率的单元作为新的单元

6. 重复第5步直到达到第2步设定的subword词表大小或概率增量低于某一阈值    

##### SentencePiece
到目前为止描述的所有标记化算法都存在相同的问题：假设输入文本使用空格来分隔单词。   
然而，并非所有语言都使用空格来分隔单词。一种可能的解决方案是使用特定于语言的预标记器language specific pre-tokenizers，例如XLM 使用特定的中文、日文和泰文预分词器）。   
为了更普遍地解决这个问题SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing (Kudo et al., 2018) ，SentencePiece：用于神经文本处理的简单且与语言无关的子词分词器和去分词器（Kudo et al., 2018）将输入视为原始输入流，从而包括要使用的字符集中的空格including the space in the set of characters to use。然后，它使用 BPE 或一元unigram 算法来构建适当的词汇表。   

例如，XLNetTokenizer 使用 SentencePiece，这也是前面示例中 "▁" 字符包含在词汇表中的原因。使用 SentencePiece 进行解码非常简单，因为所有标记都可以连接起来，并且 "▁" 被空格替换。

##### Unigram
Unigram 是 Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (Kudo, 2018) 中介绍的一种子词标记化算法。与 BPE 或 WordPiece 相比，Unigram 将其基本词汇表初始化为大量符号，并逐步修剪每个符号以获得更小的词汇表。例如，基本词汇表可以对应于所有预先标记化的单词和最常见的子串。 Unigram 不直接用于 Transformer 中的任何模型，但它与 SentencePiece 结合使用。


## EMA
指数移动平均（Exponential Moving Average，EMA）是一种用于平滑时间序列数据的方法，经常在深度学习中用于优化算法，特别是用于平滑梯度或模型参数。EMA的计算公式如下：

$ S_t = \alpha \cdot x_t + (1 - \alpha) \cdot S_{t-1} $   

通常，初始的 \( S_0 \) 可以设置为第一个输入值 \( x_0 \)。   
其中：
- \( S_t \) 是在时间 \( t \) 的EMA值。
- \( x_t \) 是在时间 \( t \) 的输入值（例如，梯度或模型参数）。
- \( \alpha \) 是平滑因子，通常取值在0到1之间。较大的 \( \alpha \) 使得EMA对当前输入更为敏感，而较小的 \( \alpha \) 使得EMA更平滑，对历史数据的影响更大。
- \( S_{t-1} \) 是在时间 \( t-1 \) 的EMA值。

在深度学习中，EMA通常用于平滑梯度或参数更新，以提高训练的稳定性。在优化算法中，EMA可以用来估计梯度的趋势，并相应地调整学习率。EMA的引入有助于减小梯度的噪声，使得训练过程更加平稳。

在代码中，对于给定的 \( \alpha \) 值，可以使用上述公式通过迭代计算来更新EMA。


## 上采样方法
hrnet 和 sd的unet 都是采用最近邻插值
下采样都使用conv            

常见的上采样方法有双线性插值、转置卷积、上采样（unsampling）和上池化（unpooling）。

其中前两种方法较为常见，后两种用得较少。

反卷积也叫转置卷积    
![alt text](assets_picture/conv_activate_token_loss/image-19.png)       





## 标签平滑（Label Smoothing）
标签平滑（Label smoothing），像L1、L2和dropout一样，是机器学习领域的一种正则化方法，通常用于分类问题，目的是防止模型在训练时过于自信地预测标签，改善泛化能力差的问题。      
Label smoothing将hard label转变成soft label    
![alt text](assets_picture/conv_activate_token_loss/image-17.png)       
![alt text](assets_picture/conv_activate_token_loss/image-18.png)    
而过大的logit差值会使模型缺乏适应性，对它的预测过于自信。在训练数据不足以覆盖所有情况下，这就会导致网络过拟合，泛化能力差，而且实际上有些标注数据不一定准确，这时候使用交叉熵损失函数作为目标函数也不一定是最优的了。      
这样，标签平滑后的分布就相当于往真实分布中加入了噪声       
Label Smoothing 劣势：       
单纯地添加随机噪音，也无法反映标签之间的关系，因此对模型的提升有限，甚至有欠拟合的风险。   
它对构建将来作为教师的网络没有用处，hard 目标训练将产生一个更好的教师神经网络。     



## 权重初始化
1. 均匀分布     
torch.nn.init.uniform_(tensor, a=0, b=1) 服从~U(a,b)      
U(a,b)

2. 正太分布     
torch.nn.init.normal_(tensor, mean=0, std=1) 服从~N(mean,std)     
N(mean,std)

3. 初始化为常数     
torch.nn.init.constant_(tensor, val) 初始化整个矩阵为常数val
       
4. Xavier
基本思想是通过网络层时，输入和输出的方差相同，包括前向传播和后向传播。具体看以下博文：

为什么需要Xavier 初始化？ 文章第一段通过sigmoid激活函数讲述了为何初始化？
简答的说就是：

如果初始化值很小，那么随着层数的传递，方差就会趋于0，此时输入值 也变得越来越小，在sigmoid上就是在0附近，接近于线性，失去了非线性
如果初始值很大，那么随着层数的传递，方差会迅速增加，此时输入值变得很大，而sigmoid在大输入值写倒数趋近于0，反向传播时会遇到梯度消失的问题
其他的激活函数同样存在相同的问题。 https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/

所以论文提出，在每一层网络保证输入和输出的方差相同。 2. xavier初始化的简单推导 https://blog.csdn.net/u011534057/article/details/51673458

对于Xavier初始化方式，pytorch提供了uniform和normal两种：

torch.nn.init.xavier_uniform_(tensor, gain=1) 均匀分布 ~ U(−a,a)
其中， a的计算公式：

torch.nn.init.xavier_normal_(tensor, gain=1) 正态分布~N(0,std)
其中std的计算公式：

5. kaiming (He initialization)
Xavier在tanh中表现的很好，但在Relu激活函数中表现的很差，所何凯明提出了针对于Relu的初始化方法。 Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification He, K. et al. (2015) 该方法基于He initialization,其简单的思想是： 在ReLU网络中，假定每一层有一半的神经元被激活，另一半为0，所以，要保持方差不变，只需要在 Xavier 的基础上再除以2

也就是说在方差推到过程中，式子左侧除以2. pytorch也提供了两个版本：

U(−bound,bound) 其中，bound的计算公式：

torch.nn.init.kaiming_normal_(tensor, a=0, mode=‘fan_in’, nonlinearity=‘leaky_relu’), 正态分布~ N(0,std)
其中，std的计算公式：

两函数的参数：

a：该层后面一层的激活函数中负的斜率(默认为ReLU，此时a=0)
mode：‘fan_in’ (default) 或者 ‘fan_out’. 使用fan_in保持weights的方差在前向传播中不变；使用fan_out保持weights的方差在反向传播中不变
针对于Relu的激活函数，基本使用He initialization，pytorch也是使用kaiming 初始化卷积层参数的。


## 量化
当然压缩方式肯定不是直接四舍五入，那样会带来巨大的精度压缩损失。常见的量化方案有 absolute-maximum 和 zero-point ，它们的差异只是 rescale 的方式不同     



## likelihood, 先验， 后验， 依据
X是特征、是属性、是对待分类物体的观测与描述；X属于{x1:有无胡须，x2：有无喉结，x3：是否穿了裙子，。。。}     
Y是分类结果；Y属于{0：男，1：女}    

先验 P（Y）：P（0）= 0.5，先于看到图片就判断分类，反映的是被分类事物的自然规律，可有多次试验用大数定律逼近；    
（事实结果）       
Evidence（依据） P（X）：P（x1=1）= 0.2，P（X）是对于各特征的一个分布，与类别Y无关，是各特征自然出现的概率（即P（x1=1）= 0.2是指，没看到此人但估计其有胡子的概率是0.2）；顾名思义，这些特征是用来进行分类的判断依据、证据；     
（事实前提）       
后验 P（Y|X）：P（0|x1=0）= 0.7，看到图片之“后”，具有图中此人所展示的这些特征的一个人是男是女的概率（P（0|x1=0）= 0.7即看到一个有胡子的这个人是男人的概率是0.7）；   
Likelihood（似然） P（X|Y）：P（x1=0|0）=0.66，被告知图中将会是一个男人，那么这个人有胡子的概率是0.66；    
(反后验)   

maximum likelihood estimation, MLE     
最大似然估计     
bert训练      

机器学习的最终目的，是学习后验概率！！！即，在训练集上学习捕捉后验概率的分布。在测试时，一个新样本输入在验视其feature之后，分析分类结果的概率，实现对分类结果的预测！！

对后验概率的直接估计是困难的。之所以称之为“贝叶斯分类器”，就是因为这里通过贝叶斯公式将对后验概率的估计转化为对likelihood * 先验 / Evidence的估计。其中，Evidence是各属性在自然界中的普世分布，当做已知。那么，对后验概率额的估计就转化为了对先验概率P（Y） 和 likelihood P(X|Y)的估计。      
P(X|Y) = P(Y|X) * P(X) / P(Y) ，即，Likelihood等于后验 * Evidence / 先验     


先验概率 P（Y）的估计：
假设样本空间各样本之间服从i.i.d （Indenpendent Identical Distribution）独立同分布，那么一句大数定律P（Y）= |Dc| / |D|，Dc是D中分类结果为c类的样本集合；

Likelihood P（X|Y）估计：有两种方法，极大似然估计（Maximum-Likelihood Estimation）和朴素贝叶斯分类器（Naive Bayes Classifier）

（1）MLE：人为猜定likelihood服从的分布形式（比如假设Likelihood服从Gaussian），然后将概率估计简化成参数估计问题。      

（2）Naive Bayes:假设所有属性xi属于x,i = 1 to d，都是条件独立的（attribute conditional independence assumption），这样一来，就不再是将每个x的一整个feature vector作为一个整体来看，而是vector中的每一个独立feature都独立地影响着分类结果。所以，对整个vector的likelihood条件概率估计就变成了对d个xi的d次条件概率估计连乘。   
？？？？       


## 度量学习 Metric Learning
度量学习 (Metric Learning) == 距离度量学习 (Distance Metric Learning，DML) == 相似度学习。  
  在数学中，一个度量（或距离函数）是一个定义集合中元素之间距离的函数。一个具有度量的集合被称为度量空间。度量学习(Metric Learning) 是人脸识别中常用的传统机器学习方法，由Eric Xing在NIPS 2002提出，可以分为两种：

通过线性变换的度量学习  
通过非线性变化的度量   

其基本原理是根据不同的任务来自主学习出针对某个特定任务的度量距离函数。后来度量学习又被迁移至文本分类领域，尤其是针对高维数据的文本处理，度量学习有很好的分类效果。  

度量学习内容

      根据不同的任务来自主学习出针对某个特定任务的度量距离函数。通过计算两张图片之间的相似度，使得输入图片被归入到相似度大的图片类别中去。

![alt text](assets_picture/conv_activate_token_loss/image-31.png)   


与经典识别网络相比

经典识别网络有一个bug：必须提前设定好类别数。 这也就意味着，每增加一个新种类，就要重新定义网络模型，并从头训练一遍。

比如我们要做一个门禁系统，每增加或减少一个员工(等于是一个新类别)，就要修改识别网络并重新训练。很明显，这种做法在某些实际运用中很不科学。

因此，Metric Learning作为经典识别网络的替代方案，可以很好地适应某些特定的图像识别场景。一种较好的做法，是丢弃经典神经网络最后的softmax层，改成直接输出一根feature vector，去特征库里面按照Metric Learning寻找最近邻的类别作为匹配项。

       目前，Metric Learning已被广泛运用于人脸识别的日常运用中。

二、为什么用度量学习？

       K-means、K近邻方法、SVM等算法，比较依赖于输入时给定的度量，比如：数据之间的相似性，那么将面临的一个基本的问题是如何获取数据之间的相似度。为了处理各种各样的特征相似度，我们可以在特定的任务通过选择合适的特征并手动构建距离函数。然而这种方法会需要很大的人工投入，也可能对数据的改变非常不鲁棒。度量学习作为一个理想的替代，可以根据不同的任务来自主学习出针对某个特定任务的度量距离函数。


在机器学习中，我们经常会遇到度量数据间距离的问题。一般来说，对于可度量的数据，我们可以直接通过欧式距离(Euclidean Distance)，向量内积(Inner Product)或者是余弦相似度(Cosine Similarity)来进行计算。  
但对于非结构化数据来说，我们却很难进行这样的操作，如计算一段视频和一首音乐的匹配程度。由于数据格式的不同，我们难以直接进行上述的向量运算，但先验知识告诉我们 ED(laugh_video, laugh_music) < ED(laugh_video, blue_music), 如何去有效得表征这种”距离”关系呢? 这就是 Metric Learning 所要研究的课题。     

Metric learning 全称是 Distance Metric Learning，它是通过机器学习的形式，根据训练数据，自动构造出一种基于特定任务的度量函数。Metric Learning 的目标是学习一个变换函数（线性非线性均可）L，将数据点从原始的向量空间映射到一个新的向量空间，在新的向量空间里相似点的距离更近，非相似点的距离更远，使得度量更符合任务的要求，如下图所示。 Deep Metric Learning，就是用深度神经网络来拟合这个变换函数。       

2. 应用
Metric Learning 技术在生活实际中应用广泛，如我们耳熟能详的人脸识别(Face Recognition)、行人重识别(Person ReID)、图像检索(Image Retrieval)、细粒度分类(Fine-grained classification)等。随着深度学习在工业实践中越来越广泛的应用，目前大家研究的方向基本都偏向于 Deep Metric Learning(DML).

一般来说, DML 包含三个部分: 特征提取网络来 map embedding, 一个采样策略来将一个 mini-batch 里的样本组合成很多个 sub-set, 最后 loss function 在每个 sub-set 上计算 loss. 如下图所示：   

![alt text](assets_picture/conv_activate_token_loss/image-32.png)  


3. 算法
Metric Learning 主要有如下两种学习范式：

3.1 Classification based:        
这是一类基于分类标签的 Metric Learning 方法。这类方法通过将每个样本分类到正确的类别中，来学习有效的特征表示，学习过程中需要每个样本的显式标签参与 Loss 计算。常见的算法有 L2-Softmax, Large-margin Softmax, Angular Softmax, NormFace, AM-Softmax, CosFace, [ArcFace](https://arxiv.org/abs/1801.07698)等。 这类方法也被称作是 proxy-based, 因为其本质上优化的是样本和一堆 proxies 之间的相似度。

3.2 Pairwise based:     
这是一类基于样本对的学习范式。他以样本对作为输入，通过直接学习样本对之间的相似度来得到有效的特征表示，常见的算法包括：Contrastive loss, Triplet loss, Lifted-Structure loss, N-pair loss, [Multi-Similarity loss](https://arxiv.org/pdf/1904.06627.pdf)等

2020 年发表的[CircleLoss](https://arxiv.org/abs/2002.10857)，从一个全新的视角统一了两种学习范式，让研究人员和从业者对 Metric Learning 问题有了更进一步的思考。     

### Deep metric learning的pipeline

深度度量学习的pipeline主要包括三个主要部分：

1.输入样本的选择和准备（data mining）；

2.网络模型结构的设计；

3.度量损失函数的设计；



deep metric leanring的核心在于最小化“相似”样本之间的某种距离度量，最大化“不相似”样本之间的某种距离度量，这里的“相似”的定义的范围非常广泛和灵活，例如对比学习中同标签也可以视为一种距离的衡量    

关于deep metric learning中的样本，只要记住：

一生二，二生三，三生万物即可。

从上文也可以看出，metric learning针对的问题主要是广义上的相似度问题，因为这里的相似可以是公式计算的，也可以是人工设计的，也可以是更加抽象的人工判定的。文中给的很多领域的应用案例，高相似度样本都是人工判定的，比如图像的相似，语义的相似等等。那么问题就转化为：

（1）我们已经有通过一些事先的方法找到的高相似的样本对，现在我们要去找低相似度的负样本对，这就涉及到negative mining的问题；

（2）我们最终要把相似度计算的问题转化为 nn训练阶段的弱监督问题。

针对于（2），其实我们可以将传统的strong supervised leanring和deep metric learning 统一到一个框架下，strong supervised learning和 weakly supervised learning，都属于supervised learning。

这是一个很有意思的想法，在传统的strong supervised learning问题中，例如lr，svm，gbdt，我们常规的思考方式是“分界面”，即线性模型学习到数据的线性分界平面，复杂的非线性模型学习到的是非线性的分界平面。另外一种思考方式是，以二分类为例，lr将原始特征进行线性变换，使得都是“1”的样本在变换后的空间聚集在一起，对于gbdt而言，则是将原始特征进行非线性变换，对于svm而言更好理解，核函数本身就是隐式的对原始特征进行映射，不同核函数对应不同的新的特征空间，然后在这个空间中进行距离计算。

而在deep metric learning中，则主要是以weakly supervised learning为主，这类supervised learning的输入不是单个样本，而往往是二元组，三元组甚至四元组。。。，无论是几元组，都可以看作是一堆的2元组构成的。

回归正题，在deep metric learning中，正样本对的数量一般是有限的，而负样本对的数量则很多时候是无限的，而不同的negtive sample pairs，对于模型训练的意义是不同的，这个层面来看和分类模型训练过程中的 sample selection问题是类似的，模型效果不佳的主要原因常常在一些hard sample的区分错误上，例如lgb中的goss tree就考虑到和easy sample和hard sample的问题，focal loss也考虑到了easy sample和hard sample的问题。而deep metric learning 在样本层面的负样本对的采样也需要考虑这个问题，良好的采样策略既能提高nn的效果，又能提高网络的训练速度。

在对比损失（contrastive loss）中确定训练样本最简单的方法是通过随机选择正或负样本对。一开始，一些论文倾向于使用简单的随机样本采样策略（随机选择两个非高相似的样本作为正负样本对）来对siamase network进行嵌入学习[29,88]。然而，[89]的作者强调，在网络达到可接受的性能水平后，学习过程可能会变慢，主要是有一些低质量的负样本对，对学习的过程完全没有什么帮助

为了解决这一问题，使用hard negative mining的方法来做采样。三态网络使用一个锚点（anchor）、一个正样本和一个负样本来训练一个网络进行分类，大概就是长（a，b，c）这样。

在[91]中，我们发现一些简单的三元组由于其判别能力较差，对模型的训练没有帮助。这些三元组造成了时间和资源的浪费。硬负样本对应的是由训练数据确定的假阳性样本。[32]中首次使用了semi-hard negative mining，目的是在给定范围内寻找负样本。与hard negative mining相比，这种采样方法采样出的负样本离锚点样本更远。


### 2.深度度量学习的损失函数和常见网络结构    
Siamese网络作为一种deep metric learning中的经典的网络结构，接收成对的图像(包括正、负样本)来训练网络模型（如下图）。成对图像之间的距离由损失函数计算。

直观上，loss function的选择会有两种：

1.和word2vec类似，构建分类问题的形式，即我们根据计算或人工判定的具有高相似的样本对的输入对应的标签为1，负样本对通过一些特定的设计的采样策略来产生（当然，人工判定的更可靠），标签为0，构建起一个二分类问题的模式（例如text match）；

2.使用确定的metric，转化为回归问题，比如余弦计算所有样本对的jaccard similarity，将jaccard similarity作为标签，这种方法使得网络学习到的结果基本上就是metric的计算结果，缺点是看起来没啥意义，直接做jaccard similarity相似度就可以完成了，优点是 把 耗时的jaccard similarity的计算转化为embedding+点积 之类的简单的计算，配合faiss等工具可以非常快速的做最近邻检索；

当然，这种方法和常规的方法之间的切换成本不高，因为我们要做的其实就是 准备好适合这种loss function的输入样本的形式，例如从 二元组输入到三元组输入，网络结构不变，改变loss function，开发成本不高的。

![alt text](assets_picture/conv_activate_token_loss/image-35.png)  
![alt text](assets_picture/conv_activate_token_loss/image-36.png)   
![alt text](assets_picture/conv_activate_token_loss/image-37.png)   
![alt text](assets_picture/conv_activate_token_loss/image-38.png)   












#### Jaccard相似度
杰卡德系数(Jaccard Index)，也称Jaccard相似系数(Jaccard similarity coefficient)，用于比较有限样本集之间的相似性与差异性。如集合间的相似性、字符串相似性、目标检测的相似性、文档查重等。   
Jaccard系数的计算方式为:交集个数和并集个数的比值:   
![alt text](assets_picture/conv_activate_token_loss/image-33.png)   
jaccard值越大说明相似度越高。   
相反地，Jaccard距离表示距离度量，用两个集合间不同样本比例来衡量:  
![alt text](assets_picture/conv_activate_token_loss/image-34.png)   

杰卡德距离用两个两个集合中不同元素占所有元素的比例来衡量两个集合的区分度。  
jaccard相似度的缺点是值适用于二元数据的集合。   

推荐算法之Jaccard相似度与Consine相似度   
对于个性化推荐来说，最核心、重要的算法是相关性度量算法。相关性从网站对象来分，可以针对商品、用户、旺铺、资讯、类目等等，从计算方式看可以分为文本相关性计算和行为相关性计算，具体的实现方法有很多种，最常用的方法有余弦夹角（Cosine）方法、杰卡德（Jaccard）方法等。Google对新闻的相似性计算采用的是余弦夹角，CBU的个性化推荐以往也主要采用此方法。从9月份开始，CBU个性化推荐团队实现了杰卡德计算方法计算文本相关性和行为相关性，并且分别在线上做了算法效果测试。本文基于测试结果，进行了对比及一些分析比较。

文本相关性的度量比较：cosine好一点点，但是Jaccard利于map/red计算
Jaccard系数主要的应用的场景

Jaccard的应用很广，最常见的应用就是求两个文档的文本相似度，通过一定的办法(比如shinging)对文档进行分词，构成词语的集合，再计算Jaccard相似度即可。当然，用途还有很多，不过大多需要结合其他的技术。比如：

过滤相似度很高的新闻，或者网页去重

考试防作弊系统

论文查重系统

计算对象间距离，用于数据聚类等。





# bn 过程

![alt text](assets/conv_activate_token_loss/image.png)

![alt text](assets/conv_activate_token_loss/image-1.png)


![alt text](assets/conv_activate_token_loss/image-3.png)

关于BN
其实BN操作的目地是使一批feature map 进行归一化，避免数据过大而导致网络性能的不稳定。我记得网有一篇博文中对BN有较详细的介绍，大概意思就是，输入数据经过若干层的网络输出其实会让数据特征分布和初始分布不一致，降低模型的泛化性能，引入BN机制后，先将特征变为标准正态分布，然后再通过γ和β两个参数将标准正态分布适当拉回归一化前的分布，相当于在原分布和标准正态分布进行折中，以此增强模型的泛化性。








一、BN过程的基本步骤
计算均值和方差：
* 对于每个mini-batch的输入数据，计算该mini-batch的均值和方差。这些均值和方差将用于对该mini-batch的输入数据进行归一化。
归一化处理：
* 使用计算得到的均值和方差，对mini-batch中的每个样本进行归一化处理，即使其均值为0，方差为1。这一步骤可以有效地减少数据的冗余性，提高模型的稳定性。
缩放和平移：
* 归一化后的数据可能会破坏原始数据的分布，因此BN过程引入了可学习的缩放因子（γ）和偏移量（β），对归一化后的数据进行线性变换，以恢复数据的表示能力。这一步骤是BN过程的关键，它允许模型在训练过程中学习到最适合当前任务的缩放和平移参数。
输出：
* 经过缩放和平移后的数据将作为下一层的输入，继续神经网络的训练过程。

二、BN过程在训练和测试中的差异
训练阶段：
* 在训练阶段，BN过程使用每个mini-batch的均值和方差进行归一化。这样可以引入一定的噪声和正则化效果，有助于模型的泛化能力。同时，由于每次迭代都使用不同的mini-batch，因此BN过程能够不断地更新均值和方差的估计，使其更加接近全局的均值和方差。
测试阶段：
* 在测试阶段，为了保持一致性和稳定性，BN过程使用训练阶段计算得到的全局均值和方差来归一化输入数据。这些全局均值和方差通常是在训练过程中通过滑动平均等方法得到的。使用全局均值和方差进行归一化可以确保测试数据与训练数据在相同的分布下进行处理，从而提高模型的性能。



![alt text](assets/conv_activate_token_loss/image-2.png)

















# 结尾
![alt text](assets_picture/conv_activate_token_loss/image-29.png)   
![alt text](assets_picture/conv_activate_token_loss/image-30.png)   