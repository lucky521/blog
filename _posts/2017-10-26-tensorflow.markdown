---
title: "TensorFlow实用手册"
categories: [design]
layout: post
---

本篇所涉及的TensorFlow API都在官方文档有所涉及，https://www.tensorflow.org/api_docs/


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

### 优化函数

```
tf.train.GradientDescentOptimizer
```


# MNIST数据集模型训练

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks

https://github.com/nlintz/TensorFlow-Tutorials/blob/master/05_convolutional_net.py

	fully_connected_feed.py 采用Feed数据方式训练。
	mnist_softmax.py  使用Softmax回归训练。
	mnist_softmax_xla.py 采用XLA框架训练。
	mnist_with_summaries.py, 05_convolutional_net.py 使用CNN训练。
	recurrent_network.py 使用RNN训练。
	autoencoder.py 使用Autoencoder训练。




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