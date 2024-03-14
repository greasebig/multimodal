ae,vae,vqvae的区别？     
! 各种iou       
clip训练过程         
图图相似度度量方法        
相同衣服识别         
tokenizer-bpe, embedding           
! lora具体怎么加         
lora训练不稳定是因为什么，怎么解决       
yolov5正负样本匹配详细规则（进而自己可以考虑为什么yolov8无锚框）        




## ae,vae,vqvae的区别？     
AE（Autoencoder）、VAE（Variational Autoencoder）和VQ-VAE（Vector Quantized Variational Autoencoder）都是神经网络中的自编码器变种，它们在实现上有一些关键区别。

AE（Autoencoder）- 自编码器：

基本原理： AE的目标是将输入数据编码成一个中间表示（编码），然后通过解码器将该中间表示重构回原始输入。
训练方式： AE通常使用均方误差（Mean Squared Error，MSE）或其他重构损失作为训练目标，通过最小化输入和重构之间的差异来学习有效的编码。
潜在空间： AE的潜在空间通常是一个连续的、光滑的空间，编码是直接映射到该空间的。
VAE（Variational Autoencoder）- 变分自编码器：

基本原理： VAE引入了潜在空间的概率分布，强制模型学习生成数据的概率分布。它通过在潜在空间中引入随机性，使得编码更加连续且有意义。
训练方式： VAE使用变分推断和重参数化技巧来训练模型，其中KL散度用于衡量学习到的潜在分布与标准正态分布之间的差异。
潜在空间： VAE的潜在空间通常是一个连续的、分布式的空间，有助于更好地捕捉数据的变化。
VQ-VAE（Vector Quantized Variational Autoencoder）- 向量量化变分自编码器：

基本原理： VQ-VAE结合了自编码器和向量量化的思想，通过向量量化将潜在表示离散化，使得编码空间更加离散且具有一定结构。
训练方式： VQ-VAE同样使用变分推断来学习潜在表示，但引入了向量量化层，通过最小化量化误差来学习离散化的表示。
潜在空间： VQ-VAE的潜在空间是离散的，通过量化实现，这意味着每个潜在表示都被映射到一组离散的码本中的一个向量。

## lora具体怎么加
lora模型在downblock.midblock,upblocks的crossattnblock中的两个attn中生效，包括toqkv,toout,都是线性映射，其中各自含有up和down的weights,bias