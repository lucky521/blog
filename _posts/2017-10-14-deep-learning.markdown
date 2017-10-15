---
title: "深度学习与神经网络"
categories: [design]
layout: post
---

这几年深度神经网络在解决模式识别问题上有很大突破，比如计算机视觉和语音识别领域。一个重要的原因是“卷积神经网络”的应用。


# 神经网络的基本构成

## Learning rate


## Activation Function


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
