---
title: "深度学习与神经网络"
categories: [design]
layout: post
---

这几年深度神经网络在解决模式识别问题上有很大突破，比如计算机视觉和语音识别领域。

一般的回归预测问题常用标准神经网络模型(Standard NN)，图片分类问题常用卷积神经网络(CNN)，语音识别问题常用递归神经网络(RNN)。


# 神经网络的基本构成

这是一个演示神经网络工作的[WEB界面](http://playground.tensorflow.org/)，通过它可以了解到神经网络的基本构成和工作流程。

![NN-playground]({{ site.baseurl }}/images/nn-playground.png)




## 神经元 Neuron

Neuron是神经网络的一个单元。它是一个函数，一个回归模型。

那么机器学习的关键就在于，怎么根据数据来训练模型，也就是怎么根据样本数据来找到最合适的模型参数。





## 代价函数 Cost Function / 损失函数 Loss / 误差函数 Error / Objective 目标函数

cost function 用于表示所训练出模型的好坏程度。它是构建一个网络模型必须的两个参数之一。

任何能够衡量模型预测出来的值 h(θ) 与真实值 y 之间的差异的函数都可以叫做代价函数 C(θ)。

比如cost function返回一个非负数，这个数会表示神经网络将训练样本映射到正确输出的准确率。cost返回值越小表明训练结果越好，那么模型训练的过程就是在使得cost尽可能小的过程。



一个好的代价函数需要满足两个最基本的要求：能够评价模型的准确性，对参数θ可微。

下面有几种常见的代价函数。


### 交叉熵 cross-entropy

对数损失函数向多类别分类问题进行的一种泛化。交叉熵可以量化两种概率分布之间的差异。

### 均方误差 Mean Square Error




## 模型的优化方法

优化方法指的是不断寻找网络的参数，使得它能显著地降低代价函数。优化器是编译网络模型必要的两个参数之一。

### SGD

### 动量

### Nesterov动量




## 数据和特征 Features of data

把真实世界的对象抽象出矩阵形式的数据，尤其是要把对结果有影响的数据以一定的数学表达抽象出来。比如图像、比如声音、比如语言。



## 隐含层 Hidden Layers

神经网络是一层一层组织起来的。每一层由一个或多个Neuron组成。

通常一个神经网络，其隐含层层数、每层的Neuron个数，都是不一定的。对于最最简单的神经网络，可以没有隐含层，那么就只有输入层和输出层，输出层有唯一一个神经元。




## 输入数据、输出数据

如果是分类问题，那么数据就是离散的值。如果是回归问题，那么数据就是连续的值。





## 激活函数 Activation Function

激活函数是在神经网络的某些层运算中加入非线性。使其能够学习更复杂的函数。如果没有它, 神经网络将只能够学习线性函数, 它只能产生输入数据的线性组合。

计算输入数据的带权之和，加上一个偏差，然后判断该样本的结果（是否“激活”）。

激活函数可以通过设置单独的激活层实现，也可以在构造层对象时通过传递activation参数实现。

下面有四种激活函数。

### Linear function

A = cx
线性函数。

### Sigmoid Function

A = 1/(1+e^(-x)

输出范围0-1之间。

### Tanh Function

A = 2/(1+e^(-2x)) - 1 = 2sigmoid(2x) - 1

输出范围是-1，1之间。


### ReLu function 线性整流函数

通常意义下，线性整流函数代指代数学中的斜坡函数，即
A = x if x>0 else 0



## 模型评估

每次轮计算后评估模型的方法。需要一个指标。



## 训练迭代

每一个pass（正向 + 反向）使用 Batch size 个样本。


### 前向传播forward pass、后向传播backward pass

正向传播比较直观，就是把一个测试数据放入当前的模型（可能还没训练完成，参数也许还是随机值），经过模型的计算，得到一个输出。这就是一次正向传播。

反向传播的目的是要用这一次正向传播的结果来更新参数。首先列出误差（正常传播结果与样本结果的误差），计算误差对某个参数的偏导函数。那么该参数的新值就更新为原值 - 更新速率 * 偏导数。


### Learning rate

Learning rate 用于表示更新参数的快慢程度。

### Epoch

Epoch 是对所有训练数据的一次 forward pass 和一次 backward pass过程。

### Batch size

Batch size 是要一次通过神经网络的样本个数。





## 正规化 Regularization method and Regularization rate

Regularization 的目的是要避免过拟合。减少真实数据生成错误（不是样本数据训练错误），避免训练数据过于片面，避免模型过于贴合训练样本而不能反映数据的真实规律。

下面有四种Regularization方法，

### Dataset augmentation

通过已有的数据集来构造新的数据，如果手里现有的样本数据不够充分、丰富，那么我们就自己人造出丰富、充分、合理的数据。

### Early stopping

在训练损失仍可以继续减少之前结束模型训练。使用早停法时，我们会在基于验证数据集的损失开始增加（也就是泛化效果变差）时结束模型训练。
迭代次数不是越多越好，我们可以在训练模型过拟合之前停止训练。
随着迭代次数增多，训练错误率会越来越小，但测试错误率可能会反弹。这就是希望我们找到一个最小化测试错误的最佳时机。

### Dropout layer

加入一层Dropout layer，在训练的时候，该层随机断开一些节点。在真正预测的时候，还是使用全部的连接。
这是正则化的一种形式，在训练神经网络方面非常有用。丢弃正则化的运作机制是，在神经网络层的一个梯度步长中移除随机选择的固定数量的单元。丢弃的单元越多，正则化效果就越强。这类似于训练神经网络以模拟较小网络的指数级规模集成学习。如需完整的详细信息，请参阅 Dropout: A Simple Way to Prevent Neural Networks from Overfitting。

### 权值衰减 Weight penalty L1 and L2

Weight penalty基于一个假设：模型的权重越小，模型越简单，要尽量使得权重的绝对值小。

Weight penalty的方法有两种：L2和L1. 他们用于附加在cost function的计算上。

L2 正规化是附加权重的平方之和，L1是附加权重的绝对值之和。





## 网络分层

### 全连接层 Fully Connected layer

所实现的运算是output = activation(dot(input, kernel)+bias)。其中activation是逐元素计算的激活函数，kernel是本层的权值矩阵，bias为偏置向量.

### Dropout层

为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。

### Activation层

激活层对一个层的输出施加激活函数。

### Flatten层

Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。


### Permute层

Permute层将输入的维度按照给定模式进行重排，例如，当需要将RNN和CNN网络连接时，可能会用到该层。


### 卷积层


### 池化层


### Recurrent层




# 神经网络实现

## 一个最简单的神经网络实现

下面是一个极度简化的神经网络实现，没有隐含层，输入是长度为3的数组，输出是一个整数。训练样本数据有四套。

激活函数使用sigmoid。要训练的模型是sigmoid(np.dot(l0,syn0))，其中l0是输入层数据，syn0是要训练的参数。

```
import numpy as np

# The simplest nerual network
# : No hidden layer

# sigmoid function
# True: f(x) = 1/(1+e^(-x))
# False: f'(x) = f(x)(1-f(x))
def sigmoid(x,deriv=False):
    if(deriv==True): # derivative function: y*(1-y)
        return x*(1-x)
    else:   # origin function
        return 1/(1+np.exp(-x))

# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights - randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(10000):
    # forward propagation
    l0 = X
    l1 = sigmoid( np.dot(l0, syn0) ) # model

    # # backward propagation
    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * sigmoid(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)


print "Output After Training:"
print l1
print "Weights after Training:"
print syn0

```


算法的关键在于，在10000次迭代中，如何更新权值参数，也就是for循环中最后三行代码。

为什么要用 "误差 * 斜率" 来更新权值参数？
这就是梯度下降吧。



# 具有隐藏层的简单神经网络实现

很多场景下，直接输入数据和输出结果是没有直观联系的，联系或者是局部的、或者是间接的，这时候我们可以加入隐藏层来发现和记录这些间接规律。

下面这一个带了一层hidden layer的神经网络。
l0是输入层， 样本数据是长度为3的数组（1X3）。
l1是隐含层，运算参数syn0是3X4矩阵。
l2是输出层，运算参数syn1是4X1的矩阵，输出结果是一个整数。

在反向传播计算误差时，某一层的误差是用后一层的"误差 * 斜率"来算的。

```
import numpy as np

# Simplest nerual network
# one hidden layer

# sigmoid function
# True: f(x) = 1/(1+e^(-x))
# False: f'(x) = f(x)(1-f(x))
def sigmoid(x,deriv=False):
    if(deriv==True): # derivative function: y*(1-y)
        return x*(1-x)
    else:   # origin function
        return 1/(1+np.exp(-x))

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

y = np.array([[0],
            [1],
            [1],
            [0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):

    # Feed forward through layers 0, 1, and 2
    l0 = X  # layer 0 - input
    l1 = sigmoid(np.dot(l0,syn0)) # layer 1 - hidden layer with syn0
    l2 = sigmoid(np.dot(l1,syn1)) # layer 2 - output with syn1

    # how much did we miss the target value?
    l2_error = y - l2

    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * sigmoid(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * sigmoid(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print
print "Weights after Training:"
print syn0
print syn1
print "Output After Training:"
print l2
```


# 有意思的东西

https://github.com/google/deepdream

https://github.com/jcjohnson/neural-style
