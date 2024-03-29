---
layout: post
title:  "分布式机器学习"
subtitle: ""
categories: [MachineLearning]
---

分布式机器学习是一个偏工程实践的领域，90%是工程、10%是数学。

# 基本概念

## 为什么需要分布式机器学习？
训练对于算力的总量要求太高了，因而穷尽各种方法想要并行。

* 计算量太大
* 训练数据太多
* 模型规模太大
  * 训练需要的内存包括模型参数、反向传播的梯度、优化器所用的内存、正向传播的中间状态（activation）


## 分布式机器学习容易遇到的问题

* 训练框架复杂化、分布式程序复杂（交互复杂、部署复杂）
* 对分布式集群资源的利用率不高（尽管对单个模型的训练速度提高了）
* 错误处理能力不好

## 怎么样的分布式？ 划分方式

* “分布式”的阶段：有训练时的分布式，有预测时的分布式（Distributed TF-Serving）。
* “分布式”的内容：有模型分布式并行，有数据分布式并行。
* “分布式”的形式：有多机器的分布式，也有单机多卡的分布式。


## 集合通信 collective communication
对于数据并行方式来讲，通信的内容就是模型的权值和训练过程中的梯度。
* broadcast，将参数从一个 node 发到多个 node 上
* reduce，将参数从多个 node 收集到一个 node 上，同时对收集到的参数进行归并(求和，求积)。
* allreduce，每个 node 都从其他 node 上面收集参数，同时对收集到的参数进行归并。







# 分布式机器学习(训练)的方式
一般有 tensor parallelism、pipeline parallelism、data parallelism 几种并行方式，分别在模型的层内、模型的层间、训练数据三个维度上对 GPU 进行划分。

## 计算并行
模型和数据都在统一一份，训练的计算资源是并行的。也就是最简单的单机多卡训练。


## 数据并行

不同的GPU有同一个模型的多个副本，每个GPU分配到不同的数据，然后将所有GPU的计算结果按照某种方式合并。

数据并行会涉及到各个GPU之间同步模型参数，一般分为同步更新和异步更新。

* 单机多GPU
* 多机多GPU
* 多机纯CPU

在目前实际应用中，单机多卡的同步式数据并行是最常用的。


## 模型并行

将机器学习模型切分为若干子模型（一部分模型参数对应于一个子模型），把每个子模型放在一个工作节点上进行计算。
子模型之间必然要有依赖关系，因此子模型划分方法关键，好的划分方法应尽可能地降低通信强度。

分布式系统中的不同GPU负责网络模型的不同部分。(例如，神经网络模型的不同网络层被分配到不同的GPU，或者同一层内部的不同参数被分配到不同GPU)


### 流水线并行 Pipeline Parallelism
流水线并行的核心思想是：在模型并行的基础上，进一步引入数据并行的办法，即把原先的数据再划分成若干个batch，送入GPU进行训练。未划分前的数据，叫mini-batch。在mini-batch上再划分的数据，叫micro-batch。

https://www.cvmart.net/community/detail/7998

https://juejin.cn/post/7262274383287484476


### 张量并行 Tensor Parallelism
对于 LLM 中的矩阵乘 法操作 Y = XA，参数矩阵 A 可以按列分成两个子矩阵 A1 和 A2，从而将原式表示为 Y = [XA1, XA2]。通过将矩阵 A1 和 A2 放置在不同的 GPU 上，矩阵乘法操作将在两个 GPU 上并行调用，并且可以通过跨 GPU 通信将两个 GPU 的输出 组合成最终结果。

https://zhuanlan.zhihu.com/p/603908668


## 3D并行
将流行线并行、张量并行和数据并行同时应用到模型训练中。

[千亿参数开源大模型 BLOOM 背后的技术](https://zhuanlan.zhihu.com/p/615839149)




# 分布式通信拓扑结构

一般来讲，分布式同步训练是通过all-reduce来完成的。分布式异步训练是通过PS服务器来完成。
当然这不意味着PS不可以做同步训练。

Parameter Server
Parameter Server的主要目的就是分布式并行进行梯度下降的计算完成参数的更新与最终收敛。

## 同步梯度更新策略 PS

ps 会同时充当 reducer 的角色，等待所有 worker 都发来梯度和参数更新请求后，ps 会对梯度取平均(reduce mean)，并用平均过后的梯度更新一次参数。各个 worker 在从 ps 读取最新参数的过程中，以及等待 ps 更新参数的过程中，都是处于空闲状态。

为了减缓因单个 ps 的有限带宽带来的阻塞，通常会设置多个 ps 对通信进行分流。

## 异步梯度更新策略 PS

parameter server异步更新策略是指每个 GPU或者CPU 计算完梯度后，无需等待其他 GPU或CPU的梯度计算（有时可以设置需要等待的梯度个数），就可立即更新整体的权值，然后同步此权值，即可进行下一轮计算.

异步的优势是速度快，缺点是worker模型更新不一致。不过实践表明，不一致带来的问题并不大，对优化目标的收敛速度影响较小。

为了实现异步，worker中pull全局参数和计算梯度是解耦进行的。

Worker和Server的交互：

- PS Worker: 
  - 从Server拉取最新模型参数
  - 利用部分训练数据，计算局部梯度
  - 上传局部梯度给Server
  
- PS Server:
  - 保存模型参数
  - 接受worker发来的局部梯度，汇总计算全局梯度，更新模型参数
  - 给worker发送新的模型参数 

分布式协作：

- 不同PS Worker如何分配任务？
  - 不同worker都是更新所有梯度。

- 多个PS Server如何工作？
  - 如果只有一个Server，它作为中心节点负载就太大了，所以引入了分布式的PS Server。
  - 每个Server只负责一部分模型参数的更新。分工方式采用一致性哈希环，按模型参数的key进行哈希分配。


[浅谈Tensorflow分布式架构：parameter server及优化策略](https://zhuanlan.zhihu.com/p/69010949)


## 同步梯度更新策略 Ring-Allreduce

AllReduce算法，是用于分布式深度学习的通信运算. 每个 GPU 只从左邻居接受数据、并发送数据给右邻居。

https://zhuanlan.zhihu.com/p/69806200

https://zhuanlan.zhihu.com/p/79030485

https://blog.csdn.net/qq_35799003/article/details/85016537


通信原语 communication primitive
* Point-to-point communication
  * 点对点通信 只有一个sender和一个receiver
* Collective communication
  * 包含多个sender多个receiver，一般的通信原语包括broadcast，gather,all-gather,scatter,reduce,all-reduce,reduce-scatter,all-to-all等


ring-base collectives



## 同步更新和异步更新的平衡



# 分布式通信拓扑框架

## MapReduce

## MPI

MPI，openMPI，openMP

- baidu-allreduce - https://github.com/baidu-research/baidu-allreduce
- NCCL - https://docs.nvidia.com/deeplearning/sdk/index.html
- rabid -  https://github.com/dmlc/rabit


## PS

基于参数服务器的分布式







# 分布式机器学习(最优化)算法

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

DistBelief（模型并行算法）

AlexNet（模型并行算法）


# 分布式机器学习推理的方式

## 分布式负载均衡
将不同推理请求路由到不同的推理机器上，达到并发的目的。

## 参数服务器
模型中Embedding词表的分布式查询。

## 单机多GPU卡推理

## 多机多GPU卡推理





# 硬件和软件协同生态
深度学习的大规模训练通常以线性增加的理想情况为基准（N个GPU应该比一个GPU快N倍）。 Horovod和NCCL库在保持高吞吐量方面做得很好，但是他们的性能与所使用的硬件有着千丝万缕的联系。高带宽和低延迟的要求导致了NVLink互连的开发。 NVIDIA DGX-2通过NVSwitch将这种互连又推进一步，该互连结构可以300GB/s的峰值双向带宽连接多达16个GPU。

## GPU间通信

- h2d
copy the data from cpu to gpu. 比如 data = data.cuda()

- 单机多卡之间通信
PCIe
Nvlink

- 多机多卡之间通信
通过 Infinteband 连接方案和 Nvidia 的 GPUDirect RDMA 技术，可以实现不同 host 上的卡间直连

## NUMA

## RDMA技术
RDMA是一种概念，在两个或者多个计算机进行通讯的时候使用DMA， 从一个主机的内存直接访问另一个主机的内存。
RDMA(RemoteDirect Memory Access)技术全称远程直接内存访问，就是为了解决网络传输中服务器端数据处理的延迟而产生的.
* ibv_post_send和ibv_post_recv原语

## Seastar

## NVIDIA-SMI Driver


## CUDA
CUDA是Nvidia GPU生态的软件基石。

NVCC: Nvidia CUDA Compiler is a proprietary compiler by Nvidia intended for use with CUDA

## NCCL
NCCL是Nvidia Collective multi-GPU Communication Library的简称.
它是一个实现多GPU的collective communication通信（all-gather, reduce, broadcast）库.
Nvidia做了很多优化，以在PCIe、Nvlink、InfiniBand上实现较高的通信速度。

- NCCL - https://docs.nvidia.com/deeplearning/sdk/index.html
- NCCL集合通信 - https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf

## cuDNN

CUDA Deep Neural Network library.

## cudatoolkit

## cuBLAS

basic linear algebra subroutines 利用cuda加速矩阵运算的库


## infiniband





# 分布式机器学习的开源工业实践

- 迭代式MapReduce  - Spark MLlib 

- 参数服务器 - Multiverso

- 数据流
  - Tensorflow  https://www.tensorflow.org/guide/distribute_strategy ,  https://www.tensorflow.org/api_docs/python/tf/distribute , https://www.tensorflow.org/guide/distributed_training

- horovod - https://eng.uber.com/horovod/
  - 控制层使用了https://www.open-mpi.org/

- Distributed (Deep) Machine Learning Community - https://github.com/dmlc

- BytePS - https://github.com/bytedance/byteps

- ray 分布式应用框架 https://github.com/ray-project/ray

- fiber - https://eng.uber.com/fiberdistributed/
  
- oneapi - 



## tf.distribute.Strategy 

tf.distribute.Strategy 是TF的高阶API中所提供的多卡、多机分布式训练的几种策略。
tf.distribute.Strategy 是 tf.estimator.RunConfig 配置入参之一。 (RunConfig是Estimator初始的入参)

训练可用tf.keras 或 tf.estimator的API， 如 estimatorAPI中的 train_and_evaluate()。

Distributed training with TensorFlow - https://www.tensorflow.org/guide/distributed_training

Multi-worker training with Estimator - https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator


https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/distribute_strategy.ipynb

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/collective_ops.py

* MirroredStrategy 适用于单机多卡，同步训练。每个节点的变量都是一致拷贝(MirroredVariable)，其内部默认使用NVIDIA NCCL来做all-reduce。
* MultiWorkerMirroredStrategy 同步训练，每个worker可以使用多个GPU。其内部实现了一个叫CollectiveOps的OP来自动选择all-reduce方法，或者自行选择(CollectiveCommunication.RING,CollectiveCommunication.NCCL)
* CentralStorageStrategy 同步训练。单机多卡。一份CPU，多份GPU。
* TPUStrategy
* ParameterServerStrategy 适用于多机多卡场景。
* OneDeviceStrategy
* CollectiveAllReduceStrategy 用于多机多卡场景，通过 all-reduce 的方式融合梯度，只需要 worker 节点，不需要 PS 节点

## mesh tensorflow 


## BytePS

BytePS是一种带有辅助带宽节点的 allreduce 实现。在使用接口上跟horovod的几乎一样。

安装时依赖numa库.

* DMLC_NUM_WORKER  worker的数量
* DMLC_NUM_SERVER  server的数量
* DMLC_PS_ROOT_URI 指定scheduler的ip地址
* DMLC_PS_ROOT_PORT 指定scheduler的端口
* NVIDIA_VISIBLE_DEVICES=0,1,2,3 允许程序看到的GPU编号
* DMLC_WORKER_ID 程序运行的GPU编号
* DMLC_ROLE 程序的角色 server/worker/scheduler
* BYTEPS_FORCE_DISTRIBUTED 调试时使用
* DMLC_INTERFACE  RDMA interface name of the scheduler

Train
```shell
# 单机多卡训练
export NVIDIA_VISIBLE_DEVICES=0,1,2,3  # gpus list
export DMLC_WORKER_ID=0 # your worker id
export DMLC_NUM_WORKER=1 # one worker
export DMLC_ROLE=worker

# the following value does not matter for non-distributed jobs
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.0.0.1
export DMLC_PS_ROOT_PORT=1234

bpslaunch python3 /usr/local/byteps/example/tensorflow/synthetic_benchmark.py --model ResNet50 --num-iters 1000000
```
介绍： https://www.zhihu.com/question/331936923
example code: https://github.com/bytedance/byteps/tree/master/example/tensorflow
使用说明：https://github.com/bytedance/byteps/blob/master/docs/step-by-step-tutorial.md
纯CPU版本的分布式训练：BytePS目前不支持纯的CPU训练。

## horovod

标准的分布式TF使用worker计算梯度，用ps平均梯度与更新参数。这样很容易产生因ps与worker分配不合理所导致的计算或通信瓶颈。

Horovod正式基于MPI实现了ring-allreduce. 无论是一个机器上的多个GPU还是多个机器上的多个GPU都可以使用。

TF+horovod : https://github.com/horovod/horovod/blob/master/docs/tensorflow.rst

* -np参数 指明使用的GPU数量
* -H参数 指明使用的GPU在什么位置
* hvd.size() 是GPU的数量
* hvd.rank() 是当前运行逻辑所在GPU的序号
* hvd.local_rank() 含义的差别rank is your index within the entire ring, local_rank is your index within your node.
* hvd.local_size() 含义 returns the number of Horovod processes within the node the current process is running on.


安装horovod需要高版本的gcc：
```shell
sudo yum install centos-release-scl
sudo yum install devtoolset-8-gcc devtoolset-8-gcc-c++
scl enable devtoolset-8 -- bash #sets gcc8 as the default compiler for a session within your current session
env HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod
```

Train
```shell
# 单机多卡训练
horovodrun -np 4 -H localhost:4 python test.py
# 多机多卡训练
horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py
```

Trace Profiler
```shell
horovodrun -np 4 --timeline-filename ./ll_timeline.json python test.py
```

example: https://github.com/horovod/horovod/tree/master/examples

纯CPU版本的分布式训练： https://github.com/horovod/horovod/issues/945



## DeepSpeed

DeepSpeed是微软开源的分布式深度学习库。

https://www.microsoft.com/en-us/research/project/deepspeed/

https://www.deepspeed.ai/getting-started/#training

ZeRO内存优化技术： 通过切片优化器状态，使得每个优化器的内存消耗减小。

ZeRO 技术旨在仅在每个 GPU 上保留部 分数据，而当需要其余数据时可以从其他 GPU 中检索。具体而言，根据三个数据部分具体的存储方式，ZeRO 提供了三 种解决方案，即优化器状态分区、梯度分区和参数分区。
PyTorch 实现了与 ZeRO 类似的技术，称为 FSDP。

* ZeRO-1
* ZeRO-2
* ZeRO-3
* ZeRO-Offload
* ZeRO-Infinity







# 参考
* [分布式训练的方案和效率对比](https://zhuanlan.zhihu.com/p/50116885)
* [分布式机器学习的论文综述](https://mp.weixin.qq.com/s/l90VsXKvcqDUvfQQe7BuRA)