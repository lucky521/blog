---
layout: post
title:  "经典模型结构"
subtitle: "如何选择适合的模型？"
categories: [MachineLearning]
---

说说久经考验的经典深度学习模型。


# LR models


# GBM models


# FM models

对categorical类型进行独热编码变成数值特征（1变多）之后，特征会非常稀疏（非零值少），特征维度空间也变大。因此FM的思路是构建新的交叉特征。

FM的表达式是在线性表达式后面加入了新的交叉项特征及对应的权值。

https://www.cnblogs.com/wkang/p/9588360.html


# Deep-learning based CTR models

https://github.com/shenweichen/DeepCTR


## Wide and Deep learning 模型

wide model (logistic regression with sparse features and transformations)
deep model (feed-forward neural network with an embedding layer and several hidden layers)


## DeepFM 模型



## Deep&Cross DCN模型

https://zhuanlan.zhihu.com/p/43364598







# Deep-learning based NLP models

## Batch Negative


## Transformer

http://jalammar.github.io/illustrated-transformer/


## Attention

https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/


## BERT 模型

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，因为decoder是不能获要预测的信息的。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。







# References

wide&deep: https://arxiv.org/pdf/1606.07792.pdf

seq2seq: https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

PNN: https://arxiv.org/pdf/1611.00144.pdf

NCF: https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf

MV-DNN: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf

FNN: https://arxiv.org/pdf/1601.02376.pdf

DNN-YouTube: https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf

DCN: https://arxiv.org/pdf/1708.05123.pdf

GBDT+LR: http://quinonero.net/Publications/predicting-clicks-facebook.pdf

FM: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

NFM: https://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-nfm.pdf

AFM: https://arxiv.org/pdf/1708.04617.pdf

deepFM: https://www.ijcai.org/proceedings/2017/0239.pdf
