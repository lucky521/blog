---
title: "深度学习与神经网络"
categories: [design]
layout: post
---

这几年深度神经网络在解决模式识别问题上有很大突破，比如计算机视觉和语音识别领域。一个重要的原因是“卷积神经网络”的应用。


# 神经网络的基本构成

## Learning rate


## 激活函数 Activation Function

计算输入数据的带权之和，加上一个偏差，然后判断该样本的结果（是否“激活”）。

下面有四种激活函数。

### Linear function 

A = cx
线性函数。

### Sigmoid Function

A = 1/(1+e^(-x)

输出范围0-1之间。

### Tanh Function

A = 2/(1+e^(-2x)) - 1 = 2sigmoid(2x) - 1

输出范围是-1，1之间


### ReLu function

A = x if x>0 else 0 


## Regularization method and Regularization rate


## Features of data


## Hidden Layers


## 输入数据、输出数据

如果是分类问题，那么数据就是离散的值。如果是回归问题，那么数据就是连续的值。


## 训练迭代

每一个pass（正向 + 反向）使用 Batch size 个样本。

### Epoch

Epoch 是对所有训练数据的一次 forward pass 和一次 backward pass过程。

### Batch size 

Batch size 是要一次通过神经网络的样本个数。




# 卷积神经网络的结构

卷积神经网络是用到了多个一样的神经元，可以表达更大的模型。

我们在写程序的时候，会写一个函数，然后在多个地方调用这个函数。这是函数的复用作用。类似的，卷积神经网络学习一个神经元，然后在多个地方复用它。


假设你有一个神经网络，训练声音样本，预测是否有人声在里面。

最简单的方法是把所有声音样本按时间平分，作为同一层的输入。复杂的方法是加入数据的其他属性，比如频率。





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
