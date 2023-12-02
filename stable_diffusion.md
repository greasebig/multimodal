# Stable Diffusion
  
  2022 SD V1.5版本  

  图像生成领域最常见生成模型有GAN和VAE，2020年，DDPM（Denoising Diffusion Probabilistic Model）被提出，被称为扩散模型（Diffusion Model），同样可用于图像生成。近年扩散模型大热，OpenAI、Google Brain等相继基于扩散模型提出的以文生图，图像生成视频生成等模型。  

## 训练数据
LAION-5B
## SD模型原理
常规的扩散模型是基于pixel的生成模型，而Latent Diffusion是基于latent的生成模型  
它先采用一个autoencoder将图像压缩到latent空间，然后用扩散模型来生成图像的latents，最后送入autoencoder的decoder模块就可以得到生成的图像。  
![Alt text](assets_picture/stable_diffusion/image.png)  
基于pixel的方法往往限于算力只生成64x64大小的图像，比如OpenAI的DALL-E2和谷歌的Imagen，然后再通过超分辨模型将图像分辨率提升至256x256和1024x1024；而基于latent的SD是在latent空间操作的，它可以直接生成256x256和512x512甚至更高分辨率的图像。

## SD模型的主体结构
autoencoder：encoder将图像压缩到latent空间，而decoder将latent解码为图像；  
CLIP text encoder：提取输入text的text embeddings，通过cross attention方式送入扩散模型的UNet中作为condition； ？？？   
UNet：扩散模型的主体，用来实现文本引导下的latent生成。  
![Alt text](assets_picture/stable_diffusion/image-1.png)  

### autoencoder
autoencoder是一个基于encoder-decoder架构的图像压缩模型，对于一个大小为$H \cdot W \cdot 3$
的输入图像，encoder模块将其编码为一个大小为$h \cdot w \cdot 3$
的latent  
f=H/h为下采样率（downsampling factor）    
除了采用L1重建损失外，还增加了感知损失（perceptual loss，即LPIPS，具体见论文The Unreasonable Effectiveness of Deep Features as a Perceptual Metric）以及基于patch的对抗训练 ??  
同时为了防止得到的latent的标准差过大，采用了两种正则化方法：第一种是KL-reg，类似VAE增加一个latent和标准正态分布的KL loss，不过这里为了保证重建效果，采用比较小的权重（～10e-6）；第二种是VQ-reg，引入一个VQ （vector quantization）layer，  

这种有损压缩肯定是对SD的生成图像质量是有一定影响的，不过好在SD模型基本上是在512x512以上分辨率下使用的。为了改善这种畸变，stabilityai在发布SD 2.0时同时发布了两个在LAION子数据集上精调的autoencoder，注意这里只精调autoencoder的decoder部分，SD的UNet在训练过程只需要encoder部分，所以这样精调后的autoencoder可以直接用在先前训练好的UNet上（这种技巧还是比较通用的，比如谷歌的Parti也是在训练好后自回归生成模型后，扩大并精调ViT-VQGAN的decoder模块来提升生成质量）  

### CLIP text encoder 
采用目前OpenAI所开源的最大CLIP模型：clip-vit-large-patch14，这个CLIP的text encoder是一个transformer模型（只有encoder模块）：层数为12，特征维度为768  
对于输入text，送入CLIP text encoder后得到最后的hidden states（即最后一个transformer block得到的特征），其特征维度大小为77x768（77是token的数量），这个细粒度的text embeddings将以cross attention的方式送入UNet中。  
训练SD的过程中，**CLIP text encoder模型是冻结的**。比如谷歌的Imagen采用纯文本模型T5 encoder来提出文本特征，而SD则采用CLIP text encoder，预训练好的模型往往已经在大规模数据集上进行了训练，它们要比直接采用一个从零训练好的模型要好。  

### UNet
其主要结构如下图所示（这里以输入的latent为64x64x4维度为例），其中encoder部分包括3个CrossAttnDownBlock2D模块和1个DownBlock2D模块，而decoder部分包括1个UpBlock2D模块和3个CrossAttnUpBlock2D模块，中间还有一个UNetMidBlock2DCrossAttn模块。encoder和decoder两个部分是完全对应的，中间存在skip connection。注意3个CrossAttnDownBlock2D模块最后均有一个2x的downsample操作，而DownBlock2D模块是不包含下采样的。  
![Alt text](assets_picture/stable_diffusion/image-2.png)   

其中CrossAttnDownBlock2D模块的主要结构如下图所示，text condition将通过CrossAttention模块嵌入进来，此时Attention的query是UNet的中间特征，而key和value则是text embeddings。(与transformer解码器第二个多头注意力层一致)  
 CrossAttnUpBlock2D模块和CrossAttnDownBlock2D模块是一致的，但是就是总层数为3。  
![Alt text](assets_picture/stable_diffusion/image-3.png)  

SD和DDPM一样采用预测noise的方法来训练UNet，其训练损失也和DDPM一样：  
![Alt text](assets_picture/stable_diffusion/image-4.png)  
这里的c 为text embeddings，此时的模型是一个条件扩散模型。  


