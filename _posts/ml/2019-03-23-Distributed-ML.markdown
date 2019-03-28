---
layout: post
title:  "分布式机器学习"
subtitle: ""
categories: [MachineLearning]
---

# 为什么需要分布式机器学习？

计算量太大

训练数据太多

模型规模太大

# 分布式机器学习的要求

数据并行

模型并行

将机器学习模型切分为若干子模型（一部分模型参数对应于一个子模型），把每个子模型放在一个工作节点上进行计算。
子模型之间必然要有依赖关系，因此子模型划分方法关键，好的划分方法应尽可能地降低通信强度。


# 通信的内容


# 分布式通信拓扑结构

迭代式MapReduce  - Spark MLlib 

参数服务器 - Multiverso

数据流 - Tensorflow



# 分布式机器学习算法

同步SGD

模型平均MA

BMUF

ADMM

弹性平均SGD

异步SGD

Hogwild!

Cylades

AdaDelay

AdaptiveRevision

带延迟补偿的异步SGD

DistBelief

AlexNet



# Distributed Tensorflow - 基于数据流的分布式机器学习

https://www.tensorflow.org/guide/distribute_strategy