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