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



# 分布式机器学习容易遇到的问题

训练框架复杂化

对分布式集群资源的利用率不高（尽管对单个模型的训练速度提高了）


# 分布式机器学习的方式

## 数据并行

不同的GPU有同一个模型的多个副本，每个GPU分配到不同的数据，然后将所有GPU的计算结果按照某种方式合并。

数据并行会涉及到各个GPU之间同步模型参数，一般分为同步更新和异步更新。

在目前实际应用中，单机多卡的同步式数据并行是最常用的。


## 模型并行

将机器学习模型切分为若干子模型（一部分模型参数对应于一个子模型），把每个子模型放在一个工作节点上进行计算。
子模型之间必然要有依赖关系，因此子模型划分方法关键，好的划分方法应尽可能地降低通信强度。

分布式系统中的不同GPU负责网络模型的不同部分。(例如，神经网络模型的不同网络层被分配到不同的GPU，或者同一层内部的不同参数被分配到不同GPU；)




# 通信的内容

对于数据并行方式来讲，通信的内容就是模型的权值和训练过程中的梯度。




# 分布式通信拓扑结构

## 异步梯度更新策略  parameter server

parameter server异步更新策略是指每个 GPU或者CPU计算完梯度后，无需等待其他 GPU或CPU的梯度计算（有时可以设置需要等待的梯度个数），就可立即更新整体的权值，然后同步此权值，即可进行下一轮计算.

https://zhuanlan.zhihu.com/p/69010949

## 同步梯度更新策略  ring all-reduce

每个 GPU 只从左邻居接受数据、并发送数据给右邻居。

https://zhuanlan.zhihu.com/p/69806200


communication primitive

    Point-to-point communication
    
    Collective communication


ring-base collectives


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







# 工业实践

迭代式MapReduce  - Spark MLlib 

参数服务器 - Multiverso

数据流 - Tensorflow  https://www.tensorflow.org/guide/distribute_strategy

horovod - https://eng.uber.com/horovod/

NCCL - https://docs.nvidia.com/deeplearning/sdk/index.html

Distributed (Deep) Machine Learning Community - https://github.com/dmlc