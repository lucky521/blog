---
title: "TensorFlow实用手册"
categories: [design]
layout: post
---

本篇所涉及的TensorFlow API都在官方文档有所涉及，https://www.tensorflow.org/api_docs/


# 重要的元素

## tf.Variable 参数

Variable 代表着模型中的参数，算法的核心目的是在训练参数。


## tf.placeholder 可变输入

在构建模型的时候没必要先把数据写好，先用tf.placeholder把数据类型确定就行。在真正session执行的时候再用feed_dict把输入填进去就好。

## tf.Session 运行数据流

在 tf.Session 之前的过程都是定义，tf.Session().run(...)才是真正执行前面定义好的操作。


## tf.summary 查看数据流图



# 常用函数


## 基础操作函数

### 基本运算函数

```
tf.reduce_mean

tf.equal

tf.argmax(vector, dimention)：返回的是vector中的最大值的索引号
```

### 类型转换函数

```
tf.cast
```

## 模型保存和加载

```
tf.train.Saver()
save_path = saver.save(sess, model_path)
saver.restore(sess, model_path)
```




## 神经网络构建函数

### 激活函数

```
tf.nn.relu
tf.nn.sigmoid
tf.nn.tanh
tf.nn.dropout
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

### 代价函数

交叉熵 softmax_cross_entropy_with_logits

softmax_cross_entropy_with_logits是用的最多的，此外还有mean_squared_error和sigmoid_cross_entropy。

```
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_nn, labels=Y)

tf.reduce_mean(cross_entropy)


```


### 优化器函数

```
tf.train.AdamOptimizer

tf.train.GradientDescentOptimizer
```




# Tensorflow机器学习模型

https://github.com/aymericdamien/TensorFlow-Examples/tree/master/examples/2_BasicModels

	kmeans
	线性回归
	逻辑回归
	KNN
	随机森林






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
