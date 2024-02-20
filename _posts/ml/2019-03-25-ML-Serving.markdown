---
layout: post
title:  "机器学习模型预测服务"
subtitle: "Model Serving/Inference"
categories: [MachineLearning]
---

机器学习技术将已有数据转变为模型，用于预测新的数据。大多数机器学习方法可以看做是学习一个函数，将输入数据（给定的trainX）映射到输出数据（给定的trainY），一旦这个函数学习完成，我们就可以用它来预测新的输入数据（TestX），从而得到我们预测的结果（TestY）。这是机器学习中学习本身的重点，可以称作是model-building。

模型预测服务，意味着我们想把预测这件事放到一个独立的节点（或模块）去处理。外界给到一条testX，制定一个model，服务就能够进行预测并传回一个所预测的testY。


# 设计原则

* 推理系统与训练系统联合。
* 性能和易用性的兼顾。
* 在所运行的设备上能高效运行。 



## 从训练环境到预测环境
Moving models from training to serving in production at scale.
一些常见的部署框架和工具包括
* TensorFlow Serving
* ONNX Runtime
* OpenVINO
* TensorRT
* TorchScript


# 模型格式
以下三类是当前常见的模型存储格式，均是把存储模型结构和模型参数分别存储。

## onnx
ONNX是一个开源的机器学习模型格式。
https://www.onnxruntime.ai/

## pytorch
pytoch参数状态字典。

## tf 
TF savedmodel。





# 模型分布式推理
将一个超大模型拆解部署在多个计算节点。（流量负载均衡、稀疏参数分布式，不在本章的讨论范围）

[?](https://github.com/tensorflow/mesh)





# 模型压缩
模型太复杂、参数太多，对于成本的要求都是很高的。因而需要模型压缩技术来尽可能权衡效果和成本。
主流的模型压缩方法，包括量化、剪枝、蒸馏、稀疏化。

## 量化(参数精度压缩)
量化是指降低模型参数的数值表示精度，比如 从 FP32 降低到 FP16 或者 INT8

训练和推理的需求不同：在训练阶段，使用高精度的浮点数可以提供更好的模型收敛性和表达能力。而在推理阶段，使用低精度可以提供更高的计算效率。因此，直接在训练过程中使用低精度可能会降低模型的准确性和性能。训练过程中的梯度计算：训练过程中需要计算梯度来更新模型参数。使用低精度表示可能导致梯度计算的不准确性，从而影响模型的收敛性和训练效果。

[大语言模型的模型量化(INT8/INT4)技术](https://zhuanlan.zhihu.com/p/627436535)

* 标准的FP32
  * 标准的 IEEE 32 位浮点表示, 为“指数”保留了 8 位，为“尾数”保留了 23 位，为符号保留了 1 位。
* 16 位浮点数 (FP16)
  * FP16 半精度浮点数，用5bit 表示指数，10bit 表示小数
* Brain Floating Point (BF16) 
  * BF16 是对FP32单精度浮点数截断数据，用8bit 表示指数，7bit 表示小数。
* int8
  * 一个 8 位的整型数据表示，可以存储 $2^8$ 个不同的值 (对于有符号整数，区间为 [-128, 127]，而对于无符号整数，区间为 [0, 255])
* 混合精度（Mixed precision）
  * 在模型中同时使用 FP32 和 FP16 的权重数值格式。 FP16 减少了一半的内存大小，但有些参数或操作符必须采用 FP32 格式才能保持准确度。
  * 比如使用 FP32 权重作为精确的 “主权重 (master weight)”，而使用 FP16/BF16 权重进行前向和后向传播计算以提高训练速度，最后在梯度更新阶段再使用 FP16/BF16 梯度更新 FP32 主权重。



* 零点量化 (zero-point quantization) 
* 最大绝对值量化 (absolute maximum quantization，absmax) 
* 对称
  * 对称量化
  * 非对称量化
* 线性
  * 线性量化
  * 非线性量化
* 饱和
  * 饱和量化
  * 非饱和量化
* 训练
  * 后训练量化（Post-Training Quantization, PTQ）
  * 量化感知训练（Quantization Aware Training, QAT）


对称量化

```python
import numpy as np
n_bit = 4
xf= np.array([0.1,0.2,1.2,3,2.1,-2.1,-3.5])
xf= np.array([15,25,35,40,50, 60, 70])
range_xf=np.max(np.abs(xf))
print('range:{}'.format(range_xf))
alpha = (2**(n_bit-1)-1)/(range_xf)
print('alpha:{}'.format(alpha))
xq=np.round(alpha*xf)
print('xq:{}'.format(xq))
de_xf=xq/alpha
print('de_xf:{}'.format(de_xf))
print('error:{}'.format(np.abs(de_xf-xf)))
print('error(sum):{}'.format(np.sum(np.abs(de_xf-xf))))
```

非对称量化

```python
import numpy as np
n_bit = 4
xf= np.array([0.1,0.2,1.2,3,2.1,-2.1,-3.5])
xf= np.array([15,25,35,40,50, 60, 70])

range_xf=np.max(xf)-np.min(xf)
print('range:{}'.format(range_xf))
alpha = (2**(n_bit-0)-1)/(range_xf)
print('alpha:{}'.format(alpha))
zp=np.round(np.min(xf)*alpha)
print('zeropoint:{}'.format(zp))
xq=np.round(alpha*xf)-zp
print('xq:{}'.format(xq))
de_xf=(xq+zp)/alpha
print('de_xf:{}'.format(de_xf))
print('error:{}'.format(np.abs(de_xf-xf)))
print('error(sum):{}'.format(np.sum(np.abs(de_xf-xf))))
```



## 参数个数压缩
复用取值相同的参数，用更少的数值表示更多的数。

## 剪枝 Weight Pruning
剪枝是指合理地利用策略删除神经网络中的部分参数，比如从单个权重到更高粒度组件如权重矩阵到通道，这种方法在视觉领域或其他较小语言模型中比较奏效。


## 蒸馏 Knowledge Distillation
蒸馏是指利用一个较小的学生模型去学习较大的老师模型中的重要信息而摒弃一些冗余信息的方法。
核心思想是通过迁移知识，从而通过训练好的大模型得到更加适合推理的小模型。


## 稀疏化
稀疏化将大量的冗余变量去除，简化模型的同时保留数据中最重要的信息。
[大语言模型的稀疏化技术](https://zhuanlan.zhihu.com/p/615399255)


## 低秩分解（Low-Rank Decomposition）
低秩分解的基本思想: 将原来大的权重矩阵分解成多个小的矩阵，用低秩矩阵近似原有权重矩阵。这样可以大大降低模型分解之后的计算量.

* SVD分解
  * np.linalg.svd
  * torch.svd_lowrank
  * tf.linalg.svd
* CP分解
* Tucker分解





# 模型编译
模型编译是将定义好的模型结构和相关参数配置转化为可执行的计算图或计算图优化的过程。
在编译阶段，模型的结构和参数被转化为底层计算库或硬件设备可执行的指令序列，以便进行高效的计算和推理。

* tvm
  * https://github.com/apache/tvm
  * pass架构： https://daobook.github.io/tvm/docs/arch/pass_infra.html
* relay
  * 可以理解为一种可以描述深度学习网络的函数式编程语言
  * Relay 是 TVM 的高级模型语言。导入到 TVM 的模型是用 Relay 表示的
  * Relay 是 TVM 中十分重要的基础组件之一，用于对接不同格式的深度学习模型以及进行模型的 transform
* mlir
  * https://mlir.llvm.org/
* iree
  * https://github.com/openxla/iree
* halide 
  * https://github.com/halide/Halide
  * 与机器学习算法无关的底层优化器，原先用于图片处理和矩阵计算

* https://github.com/alibaba/MNN
* https://github.com/Tencent/TNN
* https://github.com/bytedance/lightseq
* https://github.com/openppl-public/ppl.nn
* https://github.com/openvinotoolkit/openvino
* https://github.com/alibaba/BladeDISC



# 开源推理框架

## TensorFlow Serving
https://github.com/tensorflow/serving
TensorFlow Serving is a prediction serving system developed by Google to serve models trained in TensorFlow.
Google称它的处理能力可以达到100000 requests per second per core。


## TFlite
https://www.tensorflow.org/lite/guide


## TorchServe
https://github.com/pytorch/serve

## Clipper
https://github.com/ucbrise/clipper


## Model Server for Apache MXNet
https://github.com/awslabs/mxnet-model-server


## DeepDetect
https://github.com/jolibrain/deepdetect


## Microsoft Contextual Decision Service
Microsoft Contextual Decision Service (and accompanying paper provides a cloud-based service for optimizing decisions using multi-armed bandit algorithms and reinforcement learning, using the same kinds of explore/exploit algorithms as the Thompson sampling of LASER or the selection policies of Clipper.


## glow
https://github.com/pytorch/glow

## TVM
https://tvm.apache.org/

直接打击计算密集算子
提供了基本的图优化功能
需要人工撰写算子schedule


## MNN
MNN是阿里巴巴推出的一个高效、轻量的深度学习框架。
https://github.com/alibaba/MNN

## TNN
腾讯推出的推理引擎。

https://github.com/Tencent/ncnn

https://github.com/Tencent/TNN


## Ray Serve
https://docs.ray.io/en/latest/serve/index.html

## onnxruntime
https://github.com/microsoft/onnxruntime



# Nvidia GPU 全家桶
在模型推理方面，NVIDIA提供了基于GPU加速的推理软件。

## TensorRT (TRT)
https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
Nvidia’s TensorRT is a deep learning optimizer and runtime for accelerating deep learning inference on Nvidia GPUs.
TensorRT严格来讲并不是以一个model server框架，他的重点在于性能优化。但TensorRT提供了REST方式的服务支持。

使用上，先把TF/PyTorch模型转换为ONNX格式
* TF使用 https://github.com/onnx/tensorflow-onnx
* Pytorch使用 torch.onnx.export

```sh
python -m tf2onnx.convert \
   --input /Path/to/resnet50.pb --inputs input_1:0 \
   --outputs probs/Softmax:0 --output resnet50.onnx 
```

得到onnx格式之后，通过 trt.builder 将onnx构建出一个trt engine
```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, shape = [1,224,224,3]):

   """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file. 
      shape : Shape of the input of the ONNX file. 
  """
  with trt.Builder(TRT_LOGGER) as builder,\
    builder.create_network(1) as network, \
    builder.create_builder_config() as config, \
    trt.OnnxParser(network, TRT_LOGGER) as parser:
       config.max_workspace_size = (256 << 20)
       with open(onnx_path, 'rb') as model:
           parser.parse(model.read())
       network.get_input(0).shape = shape
       engine = builder.build_engine(network, config)
       return engine

def save_engine(engine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)

def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine
```

plan文件，该文件由trt engine序列化导出得到。 The .plan file is a serialized file format of the TensorRT engine.  

[TRT使用介绍](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/)
[b站教程视频](https://www.bilibili.com/video/BV15Y4y1W73E)


## TensorFlow-TensorRT (TF-TRT)
TensorFlow-TensorRT (TF-TRT)是一个编译器，使Tensorflow模型享受到TensorRT的加速能力。

savedmodel转换器: 
TF-TRT编译器会把一部分子图会被替换成 TRTEngineOp，这部分节点由 TensorRT 在GPU上运行。一整个模型图可能会被拆解为由若干个TensorFlow节点和若干个TensorRT共同组成。
```py
from tensorflow.python.compiler.tensorrt import trt_convert as trt
 
# Instantiate the TF-TRT converter
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir=SAVED_MODEL_DIR,
   precision_mode=trt.TrtPrecisionMode.FP32
)
 
# Convert the model into TRT compatible segments
trt_func = converter.convert()
converter.summary()
```

转换器构建之后，就可以build出trt engine
```py
MAX_BATCH_SIZE=128
def input_fn():
   batch_size = MAX_BATCH_SIZE
   x = x_test[0:batch_size, :]
   yield [x]
 
converter.build(input_fn=input_fn)
```

[TF-TRT使用介绍](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)


## Triton
https://github.com/triton-inference-server/server#readme

Triton Inference Server is an open source inference serving software that streamlines AI inferencing.
(在AI系统领域，Triton其实是个有些歧义的名字，因为至少有两个足够有影响力的Triton相关的AI系统的工作，一个是NVIDIA用于在线服务布署的Triton Inference Server，另一个是由OpenAI发起的高层次Kernel开发语言Triton。)

## FasterTransformer -> TensorRT-LLM
Nvidia的FasterTransformer是一个开源的高效Transformer层实现。
基于 CUDA, cuBLAS, cuBLASLt and C++。

https://github.com/NVIDIA/FasterTransformer
https://github.com/NVIDIA/TensorRT-LLM/











# 参考
* [A Short History of Prediction-Serving Systems](https://rise.cs.berkeley.edu/blog/a-short-history-of-prediction-serving-systems/)
* https://medium.com/@vikati/the-rise-of-the-model-servers-9395522b6c58
* https://medium.freecodecamp.org/what-we-learned-by-serving-machine-learning-models-using-aws-lambda-c70b303404a1
* https://zhuanlan.zhihu.com/p/43267451
* https://zhuanlan.zhihu.com/p/50529704
* [Dive into Deep Learning Compiler](http://tvm.d2l.ai/d2l-tvm.pdf)
