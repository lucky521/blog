---
title: "TensorFlow实用手册"
categories: [design]
layout: post
---

本篇所涉及的TensorFlow API都在官方文档有所涉及，https://www.tensorflow.org/api_docs/

# 框架体系

## TensorFlow Core

TensorFlow Core 指的是 low-level TensorFlow APIs。 https://www.tensorflow.org/guide/low_level_intro

Running the computational graph in a session

			tf.Graph
			tf.Session
			tf.placeholder


在TensorFlow中，数据以 tensor 为单元。tensor本质上是n维数组。
数组的维度叫做tensor的rank。一个标量是rank为0的tensor。
每个维度的数组长度组成的tuple元组叫做tensor的shape。


## Tensorboard

		tensorboard --logdir=/path/to/log-directory


tensorboard默认占用了6006端口

		lsof -i:6006


一个例子：https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_basic.py


Embedding Projector是Tensorboard的一个功能，可以可视化的查看embeddings。


## TensorFlow Serving

TensorFlow Serving 是基于 gRPC 和 Protocol Buffers 开发的。https://github.com/tensorflow/serving



### 服务端 tensorflow-model-server

在服务器端安装好之后，核心就是tensorflow_model_server这个binary。

		tensorflow_model_server --help

在服务端先要训练一个模型

可以用 models repo 中的例子：

		cd models/official/mnist
		python mnist.py --export_dir ./tmp/mnist_saved_model

或者用 tensorflow_serving repo中的例子：

		cd tensorflow_serving/example/
		python mnist_saved_model.py ./tmp/mnist_model


然后将这个模型载入到 TensorFlow ModelServer，注意输入的模型路径必须是绝对路径。

    tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/

https://www.tensorflow.org/serving/serving_basic


### 客户端 tensorflow-serving-api


在客户端把样本数据作为请求发送到TensorFlow ModelServer，

		python tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=localhost:9000


## TensorFlow Debugger




# 重要的元素

## tf.constant

## tf.Variable 参数

Variable 代表着模型中的参数，算法的核心目的是在训练参数。

## tf.placeholder 可变输入

在构建模型的时候没必要先把数据写好，先用tf.placeholder把数据类型确定就行。在真正session执行的时候再用feed_dict把输入填进去就好。


## tf.global_variables_initializer

## tf.Session 运行数据流

在 tf.Session 之前的过程都是定义，tf.Session().run(...)才是真正执行前面定义好的操作。

## tf.summary 查看数据流图

## tf.name_scope

## tf.variable_scope





# 常用函数


## 基础操作函数

### 基本运算函数

```
tf.random_normal

tf.random_uniform

tf.reduce_mean

tf.argmax(vector, dimention)：返回的是vector中的最大值的索引号

tf.multiply() 两个矩阵中对应元素各自相乘

tf.matmul() 将矩阵a乘以矩阵b，生成a * b。

tf.equal
```

### 类型转换函数

```
tf.cast
```

## 模型保存和加载

```
tf.train.Saver()
save_path = saver.save(sess, model_path)
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path)) # tf.train.latest_checkpoint自动获取最后一次保存的模型
saver.restore(sess, model_path)
```


## 神经网络构建函数

### 激活函数

```
tf.nn.relu
tf.nn.sigmoid
tf.nn.tanh
tf.nn.dropout
tf.nn.softmax
```

### 卷积函数

```
tf.nn.convolution
tf.nn.conv2d
```

### 池化函数

```
tf.nn.avg_pool
tf.nn.max_pool
```


## 模型训练函数

### 损失函数

交叉熵损失函数 softmax_cross_entropy_with_logits

softmax_cross_entropy_with_logits 是用的最多的，此外还有mean_squared_error和sigmoid_cross_entropy。

```
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_nn, labels=Y)

tf.reduce_mean(cross_entropy)


```


```
tf.distributions.kl_divergence
```

### 优化器函数

```
tf.train.AdamOptimizer

tf.train.GradientDescentOptimizer
```






# Tensorflow 机器学习模型

https://github.com/aymericdamien/TensorFlow-Examples/tree/master/examples/2_BasicModels

	kmeans
	线性回归
	逻辑回归
	KNN
	随机森林




# 训练 Embeddings

Embedding是一个行为，把离线形式的事物影响到为实数向量。Embedding这个词同时也是该行为所输出的东西，我们把输出的实数向量也称作是Embedding。

An embedding is a mapping from discrete objects, such as words, to vectors of real numbers.

An embedding is a relatively low-dimensional space into which you can translate high-dimensional vectors. Embeddings make it easier to do machine learning on large inputs like sparse vectors representing words. Ideally, an embedding captures some of the semantics of the input by placing semantically similar inputs close together in the embedding space. An embedding can be learned and reused across models.

这个链接讲了我们如何用TensorFlow做embedding https://www.tensorflow.org/guide/embedding，https://github.com/tensorflow/models/tree/master/tutorials/embedding

比如我们要做word embeddings.

## Do embedding

怎么把 raw format 的feature data转变为embedding format的data？

## Visualize your embeddings

把一个embedding在tensorboard上可视化出来，需要做三件事。

1) Setup a 2D tensor that holds your embedding(s).
2) Periodically save your model variables in a checkpoint in LOG_DIR.
3) (Optional) Associate metadata with your embedding.



# MNIST手写字数据集模型训练

MNIST数据集是一个手写阿拉伯数字0-9的图像素材库，它包含60000个训练样本集和10000个测试样本集。我们可以去官网下载素材库，也可以直接使用TensorFlow以package引用形式提供的MNIST。

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist


	fully_connected_feed.py 采用Feed数据方式训练。
	mnist_softmax.py  使用Softmax回归训练。
	mnist_softmax_xla.py 采用XLA框架训练。
	mnist_with_summaries.py



https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks

	autoencoder.py 使用Autoencoder训练。
	neural_network_raw.py 	2-Hidden Layers Fully Connected Neural Network
	recurrent_network.py 使用RNN训练。
	convolutional_network.py  using TensorFlow layers API
	convolutional_network_raw.py
	gan.py


https://github.com/nlintz/TensorFlow-Tutorials/blob/master/05_convolutional_net.py

	05_convolutional_net.py 使用CNN训练。





# ImageNet图像数据集模型训练

ImageNet的图像分类数据集叫做ILSVRC。ILSVRC图像分类数据集包含了来自1000个类别的120万张图片，其中每张图片属于且只属于一个类别。

## GoogLeNet

https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py

## AlexNet

https://github.com/tensorflow/models/blob/master/tutorials/image/alexnet/alexnet_benchmark.py




# LFW人脸数据集模型训练

LFW 是人脸识别数据集，英文全称是Labeled Faces in the Wild，所有人脸样本都加了标签。

https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

## FaceNet

FaceNet启发于OpenFace项目，使用TensorFlow创建的人脸识别框架。

https://github.com/davidsandberg/facenet




# 语音数据集训练模型

## spoken numbers pcm 数据集

https://github.com/pannous/tensorflow-speech-recognition

## WaveNet

https://deepmind.com/blog/wavenet-generative-model-raw-audio/
