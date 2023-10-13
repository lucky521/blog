---
layout: post
title:  "经典模型结构"
subtitle: "如何选择适合的模型？"
categories: [MachineLearning]
---

说说久经考验的机器学习经典模型。


# LR models

逻辑回归模型简单，解释性好，使用极大似然估计对训练数据进行建模。 它由两部分构成： 线性回归 + 逻辑函数

采用梯度下降法对LR进行学习训练，LR的梯度下降法迭代公式非常简洁。
LR适合离散特征，不适合特征空间大的情况。



# GBM models
xgb、catboost、RandomForest




# FM models

对categorical类型进行独热编码变成数值特征（1变多）之后，特征会非常稀疏（非零值少），特征维度空间也变大。因此FM的思路是构建新的交叉特征。

FM的表达式是在线性表达式后面加入了新的交叉项特征及对应的权值。 相比于LR， FM引入了二阶特征， 增强了模型的学习能力和表达能力。

https://www.cnblogs.com/wkang/p/9588360.html


## FFM 
Field-aware Factorization Machine









# Deep-learning based CTR models

![]({{site.baseurl}}/images/dnnmodels.jpeg)

![]({{site.baseurl}}/images/dnnmodels-2.jpg)


搜推广场景的特点:
* Sparse model
* Discrete features



## Wide and Deep learning WDL模型

可以看做是 LR + DNN

- wide model (logistic regression with sparse features and transformations) 
wide的部分具有较强的记忆能力，协同过滤、逻辑回归等简单模型的记忆能力较强。
- deep model (feed-forward neural network with an embedding layer and several hidden layers)
deep的部分具有较强的泛化能力，

## DeepFM 模型

将LR替换为FM。 可以看做是 FM + DNN

## Deep&Cross DCN 模型

它和Wide&Deep的差异就是用cross网络替代wide的部分。

Cross Layer



## DIN & DIEN

在embedding层和MLP层之间加入 attention 机制



## MOE








# Deep-learning based NLP models

## Batch Negative


## Transformer

transformer layer的样子
通过这种自注意力机制层和普通非线性层来实现对输入信号的编码，得到信号的表示。

- 图解Transformer-en http://jalammar.github.io/illustrated-transformer/
- 图解Transformer-ch https://mp.weixin.qq.com/s/g6EliR8W1AgpLm8QCcxncw
- The Annotated Transformer https://nlp.seas.harvard.edu/2018/04/03/attention.html
* 从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史 https://zhuanlan.zhihu.com/p/49271699
* 美团如何使用 Transformer 搜索排序 https://tech.meituan.com/2020/04/16/transformer-in-meituan.html
* Nvidia的FasterTransformer是一个开源的高效Transformer实现 https://github.com/NVIDIA/FasterTransformer
* 字节开源的Effective Transformer https://github.com/bytedance/effective_transformer


Transformer结构
* 把输入句子拆成词，把每个词转换为词向量，那么输入句子就变成了向量列表。
* 输入向量列表进入第一个编码器，它会把向量列表输入到 Self Attention 层，然后经过 feed-forward neural network （前馈神经网络）层，最后得到输出，传入下一个编码器。
  * Self-Attention： 
    * 对输入句子里的每一个词向量，分别和3个矩阵(WQ, WK, WV)相乘，分别得到3个新向量（Query 向量，Key 向量，Value 向量）
    * 一个词向量对应的 Query 向量和其他位置的每个词的 Key 向量的点积得分，再除以Key向量长度的开方，把这些得分的序列求softmax，再与Value向量相乘

## Attention

https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/


## BERT 模型

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，因为decoder是不能获要预测的信息的。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。

BERT 模型是最经典的编码预训练语言模型，其通过掩码语言建模和下一句预测任务，对 Transformer 模型的参数进行预训练。

https://mp.weixin.qq.com/s/mFRhp9pJRa9yHwqc98FMbg


## GPT
不再需要对于每个任务采取不同的模型架构，而是用一个取得了优异泛化能力的模型，去针对性地对下游任务进行微调。
GPT开启了”大模型“时代。




# Deep-learning based CV models

LeNet-5网络，AlexNet，VGG网络，GoogLeNet，残差网络


# GAN models

* 基于显式概率密度函数的生成模型
  * 套路：建立所需拟合的数据分布的概率密度函数，不断训练。
  * 若目标优化函数可解，直接用优化算法求解最优参数。 代表：FVBN。
  * 若目标优化函数不可解，用最近方法求解最优参数。 代表：变分近似算法（VAE）、马尔克夫蒙特卡洛算法（玻尔兹曼机）。
* 基于隐式概率密度函数的生成模型
  * 套路：直接采样的方式训练模型参数，不提前对真实数据的分布建立模型。 代表：GAN


GAN由两个子模型组成：
* 生成器G
  * G的输入是随机噪声向量
  * G的输出是与真实数据维度相同的数据
  * G的目标是生成尽可能接近真实数据的伪造数据
* 判别器D
  * D的输入是真实数据或者是G输出的伪造数据
  * D的输出是其对输入的判断分类
  * D的目标是尽可能分别出真实和伪造
* GAN模型的优化过程就是G和D的对抗过程(Generative Adversarial Nets)
  * 其中的D(..) 和 G(..) 都使用神经网络模型来拟合。
  * 当生成器G所产生的数据的概率分布与真实数据的概率分布相同时，目标函数达到最小值，参数达到最优。


训练过程：
* 训练D：
  * 从噪声输入向量Z的分布pz(z)中随机采样m个样本组成一组，输入到G；
  * 从真实数据分布pdata(x)中随机采样m个样本组成一组，直接输入到D；
  * 计算D的损失函数
  * 求解LD对θd的导数，梯度上升法更新D的参数θd
  * 反复k次
* 训练G：
  * 从pz(z)中随机采样m个样本组成一组，输入到G
  * 计算G的损失函数
  * 求解LG对θg的导数，梯度下降法更新D的参数θg
* 对于G和D的整个训练迭代N次：每次迭代都是先训练k次判别器D，再训练一次生成器G。


GAN模型的特点
* 优点
  * 没有引入近似条件和额外假设（相对于VAE）
  * 没有依赖马尔科夫链采样（相对于玻尔兹曼机）
  * 训练和推理容易并行化（相对于FVBN）
* 缺点
  * 伪数据和真实数据的分布没有交集时， 可能出现梯度消失的情况，目标无法优化
  * 超参数敏感，网络的结构设定、学习率、初始化状态等超参数对网络的训练过程影响较大，微量的超参数调整将可能导致网络的训练结果截然不同
    * DCGAN 论文作者提出了不使用Pooling 层、多使用Batch Normalization 层、不使用全连接层、生成网络中激活函数应使用 ReLU、最后一层使用tanh激活函数、判别网络激活函数应使用 LeakyLeLU 等一系列经验性的训练技巧。但是这些技巧仅能在一定程度上避免出现训练不稳定的现象，并没有从理论层面解释为什么会出现训练困难、以及如果解决训练不稳定的问题。
  * 由于约束少，容易mode collapse，难以收敛
    * 生成模型可能倾向于生成真实分布的部分区间中的少量高质量样本，以此来在判别器中获得较高的概率值，而不会学习到全部的真实分布

GAN衍生版本
* CGAN 有条件的GAN
  * 引入条件向量y，用于约束生成图像的某种属性
* LAPGAN 拉普拉斯金字塔GAN
  * 将图像的拉普拉斯金字塔和GAN相结合
* DCGAN
  * https://github.com/carpedm20/DCGAN-tensorflow
  * 深度卷积GAN
  * D和G都用卷积神经网络
  * D中池化层被带步长的卷积层替代
  * G和D的卷积层输出结果都经过BN层做归一
  * G使用ReLU，D使用Leaky ReLU
  * 优化器使用Adam
* infoGAN 互信息GAN
  * 使用GAN加上最大化生成的图片和输入编码之间的互信息
  * 输入构成：
    * 不可压缩的噪声向量z
    * 可代表明显语义特征的向量c
* LSGAN 最小二乘GAN
  * 基于最小二乘法的目标函数分别优化G和D
* WGAN
  * 采样地动距离来定义生成数据和真实数据分布之间的差异




# RNN models

* 基本 RNN 单元 

输入特点：连续输入序列，将输入x0,x1,...,xt作为从0到t时刻的序列

结构特点：有向环连接，层的输出不仅连接到下一层，还连接到自身。层内的计算不是并行而是串行的。

RNNCell: TensorFlow库中的class RNNCell是所有RNN大类模型的抽象类，实现类需要具有__call__、output_size、state_size、zero_state四个属性。

BasicRNNCell:  https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn_cell_impl.py#L238

* LSTM 单元

标准RNN的升级版：输出包含一个记忆向量，表示综合了过去时刻记忆和当前时刻输入所得的新记忆

BasicLSTMCell: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn_cell_impl.py#L361

* GRU 单元

Gated Recurrent Units的LSTM的一个变种。

GRUCell:  https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn_cell_impl.py#L272

* 双向GRU 单元

单元中包含两个隐藏层状态，它在RNN单元的基础上增加了相反方向的隐藏层状态转移。
本质上是两个独立基本RNN单元组合而成


* 外加其他特性的RNN单元

https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn_cell_impl.py

  * DropoutWrapper
    * 输入向量、隐藏层状态向量、输出向量对应的神经元分别经过Dropout层，屏蔽部分值 
  * ResidualWrapper
    * 输入可以直接与其经非线性变换后的输出组合在一起的特性
  * DeviceWrapper
    * 使某个RNN单元运行在指定设备上
  * MultiRNNCell
    * 堆叠多个RNN


## Seq2Seq






# References

wide&deep: https://arxiv.org/pdf/1606.07792.pdf

seq2seq: https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

PNN: https://arxiv.org/pdf/1611.00144.pdf

NCF: https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf

MV-DNN: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf

FNN: 利用FM的结果进行网络初始化 https://arxiv.org/pdf/1601.02376.pdf

DNN-YouTube: https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf

DCN: https://arxiv.org/pdf/1708.05123.pdf ， DCN介绍： https://zhuanlan.zhihu.com/p/43364598

GBDT+LR: http://quinonero.net/Publications/predicting-clicks-facebook.pdf

FM: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

NFM: 使用神经网络提升FM二阶部分的特征交叉能力 https://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-nfm.pdf

AFM: 引入了注意力机制的FM模型 https://arxiv.org/pdf/1708.04617.pdf

deepFM: https://www.ijcai.org/proceedings/2017/0239.pdf

深度学习推荐系统 https://www.zhihu.com/people/wang-zhe-58

CTR深度模型总结 https://github.com/shenweichen/DeepCTR

推荐系统多目标模型 https://www.cnblogs.com/eilearn/p/14746522.html