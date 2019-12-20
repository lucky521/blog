---
layout: post
title:  "机器学习模型预测服务"
subtitle: "Model Serving"
categories: [MachineLearning]
---

机器学习技术将已有数据转变为模型，用于预测新的数据。大多数机器学习方法可以看做是学习一个函数，将输入数据（给定的trainX）映射到输出数据（给定的trainY），一旦这个函数学习完成，我们就可以用它来预测新的输入数据（TestX），从而得到我们预测的结果（TestY）。这是机器学习中学习本身的重点，可以称作是model-building。

模型预测服务，意味着我们想把预测这件事放到一个独立的节点去处理。外界给到一条testX，制定一个model，服务就能够进行预测并传回一个所预测的testY。



# 从训练环境到预测环境

Moving models from training to serving in production at scale 


# 并行化

同时处理多个预测请求。




#  知名框架

## TensorFlow Serving
 https://github.com/tensorflow/serving
 TensorFlow Serving is a prediction serving system developed by Google to serve models trained in TensorFlow.
 Google称它的处理能力可以达到100000 requests per second per core。
 

## Clipper
 https://github.com/ucbrise/clipper


## Model Server for Apache MXNet
 https://github.com/awslabs/mxnet-model-server


## DeepDetect
 https://github.com/jolibrain/deepdetect


## Microsoft Contextual Decision Service
 Microsoft Contextual Decision Service (and accompanying paper provides a cloud-based service for optimizing decisions using multi-armed bandit algorithms and reinforcement learning, using the same kinds of explore/exploit algorithms as the Thompson sampling of LASER or the selection policies of Clipper.


## TensorRT
 https://github.com/NVIDIA/gpu-rest-engine
 Nvidia’s TensorRT is a deep learning optimizer and runtime for accelerating deep learning inference on Nvidia GPUs.
 TensorRT严格来讲并不是以一个model server框架，他的重点在于性能优化。但TensorRT提供了REST方式的服务支持。



# 参考

https://rise.cs.berkeley.edu/blog/a-short-history-of-prediction-serving-systems/

https://medium.com/@vikati/the-rise-of-the-model-servers-9395522b6c58

https://medium.freecodecamp.org/what-we-learned-by-serving-machine-learning-models-using-aws-lambda-c70b303404a1

https://zhuanlan.zhihu.com/p/43267451