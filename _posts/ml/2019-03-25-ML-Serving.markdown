---
layout: post
title:  "机器学习模型预测服务"
subtitle: "Model Serving/Inference"
categories: [MachineLearning]
---

机器学习技术将已有数据转变为模型，用于预测新的数据。大多数机器学习方法可以看做是学习一个函数，将输入数据（给定的trainX）映射到输出数据（给定的trainY），一旦这个函数学习完成，我们就可以用它来预测新的输入数据（TestX），从而得到我们预测的结果（TestY）。这是机器学习中学习本身的重点，可以称作是model-building。

模型预测服务，意味着我们想把预测这件事放到一个独立的节点（或模块）去处理。外界给到一条testX，制定一个model，服务就能够进行预测并传回一个所预测的testY。


# 设计原则

* 与训练系统联合。
* 性能和易用性的兼顾。
* 在所运行的设备上能高效运行。 



# 从训练环境到预测环境
Moving models from training to serving in production at scale.
一些常见的部署框架和工具包括 TensorFlow Serving、ONNX Runtime、OpenVINO、TensorRT、TorchScript.


# 模型格式

ONNX是一个开源的机器学习模型格式。
https://www.onnxruntime.ai/


pytoch参数状态字典。


TF savedmodel。





# 模型分布式
将一个超大模型拆解部署在多个节点。

https://github.com/tensorflow/mesh

# 模型压缩
模型太复杂、参数太多，对于成本的要求都是很高的。因而需要模型压缩技术来尽可能权衡效果和成本。
主流的模型压缩方法，包括量化、剪枝、蒸馏、稀疏化。

## 量化
量化是指降低模型参数的数值表示精度，比如 从 FP32 降低到 FP16 或者 INT8

训练和推理的需求不同：在训练阶段，使用高精度的浮点数可以提供更好的模型收敛性和表达能力。而在推理阶段，使用低精度可以提供更高的计算效率。因此，直接在训练过程中使用低精度可能会降低模型的准确性和性能。训练过程中的梯度计算：训练过程中需要计算梯度来更新模型参数。使用低精度表示可能导致梯度计算的不准确性，从而影响模型的收敛性和训练效果。

[大语言模型的模型量化(INT8/INT4)技术](https://zhuanlan.zhihu.com/p/627436535)

* 16 位浮点数(FP16)
  * FP16半精度浮点数，用5bit 表示指数，10bit 表示小数
* Brain Floating Point (BF16) 
  * BF16 是对FP32单精度浮点数截断数据，即用8bit 表示指数，7bit 表示小数。
* int8



## 剪枝
剪枝是指合理地利用策略删除神经网络中的部分参数，比如从单个权重到更高粒度组件如权重矩阵到通道，这种方法在视觉领域或其他较小语言模型中比较奏效。

## 蒸馏
蒸馏是指利用一个较小的学生模型去学习较大的老师模型中的重要信息而摒弃一些冗余信息的方法。
核心思想是通过迁移知识，从而通过训练好的大模型得到更加适合推理的小模型。

## 稀疏化
稀疏化将大量的冗余变量去除，简化模型的同时保留数据中最重要的信息。
[大语言模型的稀疏化技术](https://zhuanlan.zhihu.com/p/615399255)


# 模型编译
模型编译是将定义好的模型结构和相关参数配置转化为可执行的计算图或计算图优化的过程。在编译阶段，模型的结构和参数被转化为底层计算库或硬件设备可执行的指令序列，以便进行高效的计算和推理。

* halide 
  * 与机器学习算法无关的底层优化器，原先用于图片处理和矩阵计算

* relay
  * 可以理解为一种可以描述深度学习网络的函数式编程语言


* https://github.com/alibaba/MNN
* https://github.com/Tencent/TNN
* https://github.com/bytedance/lightseq
* https://github.com/microsoft/onnxruntime
* https://github.com/apache/tvm
* https://github.com/openppl-public/ppl.nn
* https://github.com/openvinotoolkit/openvino
* https://github.com/alibaba/BladeDISC



# 知名框架

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




# Nvidia GPU 全家桶

## TensorRT (TRT)
https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
Nvidia’s TensorRT is a deep learning optimizer and runtime for accelerating deep learning inference on Nvidia GPUs.
TensorRT严格来讲并不是以一个model server框架，他的重点在于性能优化。但TensorRT提供了REST方式的服务支持。

使用上，先把TF/PyTorch模型转换为ONNX格式（使用https://github.com/onnx/tensorflow-onnx）

```py
python -m tf2onnx.convert  --input /Path/to/resnet50.pb --inputs input_1:0 --outputs probs/Softmax:0 --output resnet50.onnx 
```

得到onnx格式之后，通过trt.builder将onnx构建出一个trt engine
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
   with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
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


## triton
https://github.com/triton-inference-server/server#readme

Triton Inference Server is an open source inference serving software that streamlines AI inferencing.
(在AI系统领域，Triton其实是个有些歧义的名字，因为至少有两个足够有影响力的Triton相关的AI系统的工作，一个是NVIDIA用于在线服务布署的Triton Inference Server，另一个是由OpenAI发起的高层次Kernel开发语言Triton。)

## FasterTransformer
Nvidia的FasterTransformer是一个开源的高效Transformer实现 https://github.com/NVIDIA/FasterTransformer



# 参考
* [A Short History of Prediction-Serving Systems](https://rise.cs.berkeley.edu/blog/a-short-history-of-prediction-serving-systems/)
* https://medium.com/@vikati/the-rise-of-the-model-servers-9395522b6c58
* https://medium.freecodecamp.org/what-we-learned-by-serving-machine-learning-models-using-aws-lambda-c70b303404a1
* https://zhuanlan.zhihu.com/p/43267451
* https://zhuanlan.zhihu.com/p/50529704
* [Dive into Deep Learning Compiler](http://tvm.d2l.ai/d2l-tvm.pdf)
