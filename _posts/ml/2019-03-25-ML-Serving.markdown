---
layout: post
title:  "机器学习模型推理服务"
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
* ONNXRuntime
* OpenVINO
* TensorRT
* TorchScript


# 模型格式
模型存储格式对模型逻辑(结构+参数)的一种中间表示。类比:tensorflow/pytorch的python程序是高级语言； runtime硬件执行的程序是低级语言。
以下三类是当前常见的模型存储格式，均是把存储模型结构和模型参数分别存储。

## onnx
ONNX是一个开源的机器学习模型格式。
onnx文件是一个ModelProto，它包含了一些版本信息，生产者信息和一个GraphProto。在GraphProto里面又包含了四个repeated数组，它们分别是node(NodeProto类型)，input(ValueInfoProto类型)，output(ValueInfoProto类型)和initializer(TensorProto类型)，其中node中存放了模型中所有的计算节点，input存放了模型的输入节点，output存放了模型中所有的输出节点，initializer存放了模型的所有权重参数。


onnx opset 


## pytorch
pytoch参数状态字典。

## tf 
TF savedmodel。

## safetensors
huggingface自研的模型文件格式，主打安全和无多余内存占用。
* 前面的 8 bytes是一个无符号的整数，表示 header 占的字节数。
* 中间的 N bytes是一个UTF-8编码JSON字符串，存储 header 的内容，里面为模型权重的元数据信息。
* 文件的剩余部分存储模型权重 tensor 的值。


## gguf
GGUF文件全称是GPT-Generated Unified Format，是由Georgi Gerganov定义发布的一种大模型文件格式。Georgi Gerganov是著名开源项目llama.cpp的创始人。





# 模型分布式推理
将一个超大模型的计算部分拆解，部署在多个计算节点。（流量负载均衡、稀疏参数的分布式化，不在本章的讨论范围）
这是一个很大的话题，见另一篇以分布式机器学习为主题的blog。
[?](https://github.com/tensorflow/mesh)





# 模型压缩 (model compression and acceleration)
模型太复杂、参数太多，对于成本的要求都是很高的。因而需要模型压缩技术来尽可能权衡效果和成本。
主流的模型压缩方法，包括量化、剪枝、蒸馏、稀疏化。

该领域的知名学者 https://hanlab.mit.edu/songhan

模型压缩的动机一般有两类， 一类是存储太大，模型存不下；一类是计算太多，计算太慢；对于具体场景，你应该明确自己的短板在哪儿，主诉是什么，才好选出适合你的方法。

## 精度量化(参数精度压缩)
量化是指降低模型参数的数值表示精度，比如 从 FP32 降低到 FP16 或者 INT8.
***说简单点：模型量化过程其实就是在做一件事，就是找阈值或者scale。***

训练和推理的需求不同：在训练阶段，使用高精度的浮点数可以提供更好的模型收敛性和表达能力。而在推理阶段，使用低精度可以提供更高的计算效率。因此，直接在训练过程中使用低精度可能会降低模型的准确性和性能。训练过程中的梯度计算：训练过程中需要计算梯度来更新模型参数。使用低精度表示可能导致梯度计算的不准确性，从而影响模型的收敛性和训练效果。

### 有哪些低精度表示

* 32位标准浮点数 FP32
  * 标准的 IEEE 32 位浮点表示, 为“指数”保留了 8 位，为“尾数”保留了 23 位，为符号保留了 1 位。
* TensorFloat TF32
  * Nvidia A系列开始支持的跟fp32相同范围的TensorCore数据格式，性能比fp32 cuda core快8倍；
* 16位浮点数 (FP16)
  * FP16 半精度浮点数，用5bit 表示指数，10bit 表示小数
* Brain Floating Point (BF16) 
  * BF16 是对FP32单精度浮点数截断数据，用8bit 表示指数，7bit 表示小数。
* int8
  * 一个 8 位的整型数据表示，可以存储 $2^8$ 个不同的值 (对于有符号整数，区间为 [-128, 127]，而对于无符号整数，区间为 [0, 255])
* fp8
  * FP8采用两种表示方式，分别是E4M3和E5M2，其中E代表指数位（Exponent），M代表尾数位（Mantissa）。在特定的表示范围内，E4M3展现出了更为精确的数值表现，而E5M2则以其更宽广的动态范围见长。
* nf4
  * 更极限，4-bit表示
* 混合精度（Mixed precision, fp16&fp32）
  * 在模型中同时使用 FP32 和 FP16 的权重数值格式。 FP16 减少了一半的内存大小，但有些参数或操作符必须采用 FP32 格式才能保持准确度。
  * 比如使用 FP32 权重作为精确的 “主权重 (master weight)”，而使用 FP16/BF16 权重进行前向和后向传播计算以提高训练速度，最后在梯度更新阶段再使用 FP16/BF16 梯度更新 FP32 主权重。

fp16 vs bf16: fp16的精度相对高、数值范围相对小、支持的机型相对多、速度相对慢
fp32 vs tf32: tf32的精度相对低、性能相对高


### 精度量化的几大分类
量化的本质是利用scale+zero_point把一个范围的值都映射到另一个范围里；以下无论是什么方法，最终都是找到scale+zero之后去映射。
***如何减轻数据从浮点转换到定点的精度损失，怎么样Quantization后和de-quantization差异最小化，是整个量化研究的重点.***

* 零点量化 (zero-point quantization) 
  * q = round(r/s)127
* 最大绝对值量化 (absolute maximum quantization，absmax) 
* 对称
  * 对称量化
  * 非对称量化
* 线性
  * 线性量化
  * 非线性量化
* 饱和
  * 饱和量化 ： 原始分布不均匀的时候采用，为了映射到新阈值之后的分布相对均匀。 不再是基于最大值进行缩放, 而是找一个阈值|T|,关键是如何找到最优的阈值。
  * 非饱和量化:  原始分布均匀的时候采用。
* 训练
  * 后训练量化（Post-Training Quantization, PTQ）
    * 不带校准的PTQ
    * 带校准的PTQ: 用一个所谓校准集的东西来进行抽样分布，有了抽样分布后再做量化阈值的选取
  * 量化感知训练（Quantization Aware Training, QAT）
    * 利用一个量化过程 q = round(r/s)127，将需要量化的值量化到0, 127之间，再接着一个反量化过程q  s，就实现了一个误差的传递。通过训练学习量化参数q。


* 指数平滑法，
  * 即将校准数据集送入模型，收集每个量化层的输出特征图，计算每个batch的S和Z值，并通过指数平滑法来更新S和Z值。
* 直方图截断法，
  * 即在计算量化参数Z和S的过程中，由于有的特征图会出现偏离较远的奇异值，导致max非常大，所以可以通过直方图截取的形式，比如抛弃最大的前1%数据，以前1%分界点的数值作为max计算量化参数。
* KL散度校准法，
  * 即通过计算KL散度（也称为相对熵，用以描述两个分布之间的差异）来评估量化前后的两个分布之间存在的差异，搜索并选取KL散度最小的量化参数Z和S作为最终的结果。TensorRT中就采用了这种方法。



### 量化的模型哪些部位
量化的参数主要分为2类: 权重和激活值(特征)
* 待量化 Op 的权重：  一般分布均匀
* 待量化 Op 的激活值(特征)Tensor（包括输入和输出）: 一般分布不均匀



### 量化公式代码实例

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


### 量化工具
* onnxruntime自带工具
  * https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
  * https://onnxruntime.ai/docs/performance/model-optimizations/float16.html
* pytorch自带工具
  * torch.quantization
  * https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization
* huggingface自带工具
  * https://huggingface.co/docs/transformers/en/quantization
* tensorrt-llm中的量化
  * https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/README.md


### QA
Q:如何理解对weights(权值)进行量化？


Q:如何理解对activations(激活)进行量化?


Q:为什么有人说对激活量化对精度效果影响很大？


Q: llm.int8 所谓的 "绝大部分权重和激活用8bit量化，对离群特征的几个维度保留16bit，进行高精度的矩阵乘法。"  如何识别离群特征？
https://arxiv.org/pdf/2208.07339
https://fancyerii.github.io/2024/01/16/int8/


Q: 评估精度的方法？
* 直接比较结果的diff
* 比较 auc


## 参数个数压缩 (weight sharing)
复用取值相同的参数，用更少的数值表示更多的数。

## 剪枝 (Weight Pruning)
剪枝是指合理地利用策略删除神经网络中的部分参数，比如从单个权重到更高粒度组件如权重矩阵到通道，这种方法在视觉领域或其他较小语言模型中比较奏效。

综述： https://zhuanlan.zhihu.com/p/692858636

OBD -> OBS 算法：
它的初衷就是如何选择性的删除一些权重从而减小网络大小，但别引入太多的误差？
Optimal Brain Damage: 
Optimal Brain Surgeon

## 蒸馏 (Knowledge Distillation)
蒸馏是指利用一个较小的学生模型去学习较大的老师模型中的重要信息而摒弃一些冗余信息的方法。
核心思想是通过迁移知识，从而通过训练好的大模型得到更加适合推理的小模型。


## 稀疏化
稀疏化将大量的冗余变量去除，简化模型的同时保留数据中最重要的信息。
[大语言模型的稀疏化技术](https://zhuanlan.zhihu.com/p/615399255)


## 低秩分解（Low-Rank Decomposition,  low-rank factorization）

当矩阵的秩较低时（r << n, m），就可以视其为低秩矩阵。低秩矩阵意味着，此矩阵中有较多的行（或列）是线性相关的，即：信息冗余较大。

低秩分解的基本思想: 将原来大的权重矩阵分解成多个小的矩阵，用低秩矩阵近似原有权重矩阵。这样可以大大降低模型分解之后的计算量.

* SVD分解
  * np.linalg.svd
  * torch.svd_lowrank
  * tf.linalg.svd
* CP分解
* Tucker分解


## early exit
推理的时候提前结束，以优化推理速度。
* Confidence Estimation
* Internal Ensemble
* Learning to Exit


## token skipping
* PoWER-BERT (Goyal等，2020)：PoWER-BERT通过在每个Transformer层之间丢弃一部分token来实现加速，这一过程基于每个token接收到的注意力。每层需要丢弃的token数量（即，计划）是通过与原始损失函数共同优化一个软掩码层的稀疏性来学习的。这种方法在准确率-时间权衡的Pareto曲线上获得了更好的结果。
* TR-BERT (Ye等，2021)：TR-BERT引入了一个动态机制来决定跳过哪些token。它使用奖励机制进行强化学习训练，该奖励既促进分类器的置信度，又惩罚保留token的数量。与PoWER-BERT不同，被跳过的token会被转发到最后一层，而不是被移除。
* Length-Adaptive Transformer (LAT, Kim和Cho，2021)：LAT引入了LengthDrop，它在预训练期间随机跳过token，以减少预训练和微调之间的差距。LAT的计划是通过进化搜索算法来搜索的。
* LTP (Kim等，2022)：LTP为每个Transformer层学习一个阈值。与遵循计划丢弃特定数量的token不同，LTP简单地丢弃那些具有低于学习阈值的显著性得分（接收到的注意力）的token。
* Transkimmer (Guan等，2022)：Transkimmer在每一层之前添加了一个小型的多层感知器（MLP）和Gumbel-Softmax重参数化组成的skim预测器模块。这些skim预测器输出一个决定是否丢弃token的掩码。它还采用了一种skim损失，该损失优化跳过的token与总token数量的比例，以鼓励稀疏性。





# 模型编译
模型编译是将定义好的模型结构和相关参数配置转化为可执行的计算图或计算图优化的过程。
在编译阶段，模型的结构和参数被转化为底层计算库或硬件设备可执行的指令序列，以便进行高效的计算和推理。

## 模型编译器的有关概念
许多概念可以从这里学习： https://openmlsys.github.io/
* 机器学习框架前端
  *  PyTorch、TensorFlow 和 JAX
* 定义IR计算图
  * 计算图由基本数据结构：张量(Tensor)和基本运算单元：算子(Operator)构成
  * 正确处理张量数据类型
* AI编译器前端
  * 把源程序解析成IR，做各种和硬件无关的优化

## AI编译器重点项目

* tvm
  * https://github.com/apache/tvm
  * pass架构： https://daobook.github.io/tvm/docs/arch/pass_infra.html
* relay
  * 可以理解为一种可以描述深度学习网络的函数式编程语言
  * Relay 是 TVM 的高级模型语言。导入到 TVM 的模型是用 Relay 表示的
  * Relay 是 TVM 中十分重要的基础组件之一，用于对接不同格式的深度学习模型以及进行模型的 transform
* mlir
  * https://mlir.llvm.org/
* xla
  * https://github.com/openxla/xla
* iree
  * https://github.com/openxla/iree
* halide 
  * https://github.com/halide/Halide
  * 与机器学习算法无关的底层优化器，原先用于图片处理和矩阵计算
* BladeDISC
  * https://github.com/alibaba/BladeDISC
  * [BladeDISC++：Dynamic Shape AI 编译器下的显存优化技术](https://zhuanlan.zhihu.com/p/18880631601)
* byteIR
  * https://github.com/bytedance/byteir
* PPLNN
  * https://github.com/openppl-public/ppl.nn





# 开源推理框架

## onnxruntime
https://github.com/microsoft/onnxruntime
微软推出的一种模型标准，它设计了一种模型文件的表达结构，又实现了一个跨平台的执行引擎，用于在各种硬件设备上执行这个模型的推理过程。

* libonnxruntime.so
* libonnxruntime_providers_shared.so
* libonnxruntime_providers_*.so

## openvino
* https://github.com/openvinotoolkit/openvino
不足之处可能是只支持 Intel 家的硬件产品。


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


## 其他

* https://github.com/bytedance/lightseq






# Nvidia GPU 加速推理全家桶
在模型推理方面，NVIDIA提供了基于GPU加速的推理软件。

## TensorRT (TRT)
* [入口文档](https://docs.nvidia.com/deeplearning/tensorrt/index.html)
* [TRT使用介绍](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/)
* [b站教程视频](https://www.bilibili.com/video/BV15Y4y1W73E)
* [hello-world](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-803/quick-start-guide/index.html#run-engine-c)
* [OP](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/index.html)
* [trtexec用法](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)
* [性能优化](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-803/best-practices/index.html)

Nvidia’s TensorRT is a deep learning optimizer and runtime for accelerating deep learning inference on Nvidia GPUs.
TensorRT严格来讲并不是以一个model server框架，他的重点在于性能优化。但TensorRT提供了REST方式的服务支持。TensorRT中的profile指的是优化推理的配置，它定义了一组输入张量的形状范围，并允许TensorRT为这些范围内的各种形状生成高效的推理引擎。


概念上，
1. profile 是一个重要的概念，特别是在处理动态形状（dynamic shapes）和优化推理性能时





使用上，
1.先把TF/PyTorch模型转换为ONNX格式
* TF使用 https://github.com/onnx/tensorflow-onnx -> tf2onnx.convert
* Pytorch使用 torch.onnx.export

```sh
python -m tf2onnx.convert \
   --input /Path/to/resnet50.pb --inputs input_1:0 \
   --outputs probs/Softmax:0 --output resnet50.onnx 
```

2.得到onnx格式之后，通过 trt.builder 将onnx构建出一个trt engine
(plan文件，该文件由trt engine序列化导出得到。 The .plan file is a serialized file format of the TensorRT engine.)

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

上面是python用法，下面看看c++接口

deserializeCudaEngine




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


## trtexec
https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec
这是一个命令行工具， 可以根据onnx模型文件生成tensorrt引擎文件; 可以直接加载tensorrt引擎文件做推理；

## Triton
https://github.com/triton-inference-server/server#readme

Triton Inference Server is an open source inference serving software that streamlines AI inferencing.
(在AI系统领域，Triton其实是个有些歧义的名字，因为至少有两个足够有影响力的Triton相关的AI系统的工作，一个是NVIDIA用于在线服务布署的Triton Inference Server，另一个是由OpenAI发起的高层次Kernel开发语言Triton。)

## FasterTransformer -> TensorRT-LLM
Nvidia的FasterTransformer是一个开源的高效Transformer层实现。
基于 CUDA, cuBLAS, cuBLASLt and C++。

https://github.com/NVIDIA/FasterTransformer -> https://github.com/NVIDIA/TensorRT-LLM/











# 参考
* [A Short History of Prediction-Serving Systems](https://rise.cs.berkeley.edu/blog/a-short-history-of-prediction-serving-systems/)
* https://medium.com/@vikati/the-rise-of-the-model-servers-9395522b6c58
* https://medium.freecodecamp.org/what-we-learned-by-serving-machine-learning-models-using-aws-lambda-c70b303404a1
* https://zhuanlan.zhihu.com/p/43267451
* https://zhuanlan.zhihu.com/p/50529704
* [Dive into Deep Learning Compiler](http://tvm.d2l.ai/d2l-tvm.pdf)
* [大语言模型的模型量化(INT8/INT4)技术](https://zhuanlan.zhihu.com/p/627436535)