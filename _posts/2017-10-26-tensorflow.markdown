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


## 神经网络相关函数

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

## 代价函数

```
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_nn, labels=Y))

softmax_cross_entropy_with_logits是用的最多的，此外还有mean_squared_error和sigmoid_cross_entropy。

```


### 优化函数

```
tf.train.GradientDescentOptimizer
```



# MNIST手写字数据集模型训练

MNIST数据集是一个手写阿拉伯数字0-9的图像素材库，它包含60000个训练样本集和10000个测试样本集。我们可以去官网下载素材库，也可以直接使用TensorFlow以package引用形式提供的MNIST。

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist


	fully_connected_feed.py 采用Feed数据方式训练。
	mnist_softmax.py  使用Softmax回归训练。
	mnist_softmax_xla.py 采用XLA框架训练。
	mnist_with_summaries.py
	autoencoder.py 使用Autoencoder训练。


https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks

	neural_network_raw.py 	2-Hidden Layers Fully Connected Neural Network
	recurrent_network.py 使用RNN训练。
	convolutional_network.py
	convolutional_network_raw.py


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

https://github.com/davidsandberg/facenet

## Openface

https://github.com/cmusatyalab/openface





# 语音数据集训练模型

## spoken numbers pcm 数据集

https://github.com/pannous/tensorflow-speech-recognition



<!--
这里是注释区

```
print "hello"
```

***Stronger***

{% highlight python %}
print "hello, Lucky!"
{% endhighlight %}

![My image]({{ site.baseurl }}/images/emule.png)

My Github is [here][mygithub].
[mygithub]: https://github.com/lucky521

-->