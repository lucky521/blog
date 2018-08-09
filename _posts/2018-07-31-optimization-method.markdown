---
title: "机器学习的最优化方法"
subtitle: "Optimization Method"
categories: [design]
layout: post
---

机器学习里模型训练的过程是在反复优化一个函数，使得其能做出理想的预测。而优化时又会设计一个Objective Function，用来衡量模型预测出来的值 h(θ) 与真实值 y 之间的差异。当然，差异越小且越稳定越好。我们所说的最优化方法，就是要优化这个Objective Function，使其取得一个极小值。

# 目标函数和目标模型

这里需要考虑几个问题。

## 如何确定满足最小化Objective Function就能训练出的好模型？


## 如何保证和确定当前的Objective Function是可优化的？


## 如何确定当前的Objective Function已取得极小值？



# Objective Function的设定

目标函数的设定中要包含loss function 和 regularization。前者是要使得我们的模型尽可能的学习训练样本的分布。后者是希望模型不用过拟合。

Training Loss measures how well model fit on training data.

Regularization measures complexity of model.


## Regressive loss functions

They are used in case of regressive problems, that is when the target variable is continuous.
Most widely used regressive loss function is Mean Square Error. Other loss functions are:
1. Absolute error — measures the mean absolute value of the element-wise difference between input;
2. Smooth Absolute Error — a smooth version of Abs Criterion.


## Classification loss functions

The output variable in classification problem is usually a probability value f(x), called the score for the input x. Generally, the magnitude of the score represents the confidence of our prediction. The target variable y, is a binary variable, 1 for true and -1 for false.
On an example (x,y), the margin is defined as yf(x). The margin is a measure of how correct we are. Most classification losses mainly aim to maximize the margin. Some classification algorithms are:
1. Binary Cross Entropy
2. Negative Log Likelihood
3. Margin Classifier
4. Soft Margin Classifier


## Embedding loss functions

It deals with problems where we have to measure whether two inputs are similar or dissimilar. Some examples are:
1. L1 Hinge Error- Calculates the L1 distance between two inputs.
2. Cosine Error- Cosine distance between two inputs.


## Regularization

0. L0 Regularization

L0范数是指向量中非0的元素的个数。

1. L1 Regularization

L1范数是指向量中各个元素绝对值之和，也叫“稀疏规则算子”（Lasso regularization）。

2. L2 Regularization

 L2范数是指向量各元素的平方和然后求平方根。



# 约束优化问题

## 无约束的优化问题

无约束最优化问题是优化领域研究的基础性问题。

常见的求解方法有梯度下降法、最速下降法、牛顿法、共轭梯度法（DFP法）和变尺度法（变度量法）。

对于特殊的最小二乘问题，即把均方误差作为最小化目标，有最小二乘法。

## 有等式约束的优化问题

使用拉格朗日乘子法（Lagrange Multiplier) 求解。

常常丽勇拉格朗日对偶性将原始问题转换为对偶问题，通过解对偶问题而得到原始问题的解。

## 有不等式约束的优化问题

使用KKT条件求解。

[参考](https://zhuanlan.zhihu.com/p/26514613)



# 最优化方法

常见的有三种梯度下降方法，主要区别是在使用多少数据来计算objective function的梯度。

## Batch gradient descent

θ=θ−η⋅∇θJ(θ)

```
for i in range(nb_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad
```

## Stochastic gradient descent

θ=θ−η⋅∇θJ(θ;x(i);y(i))

```
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
```

## Mini-batch gradient descent

θ=θ−η⋅∇θJ(θ;x(i:i+n);y(i:i+n))

```
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
```

## Momentum

## Nesterov accelerated gradient

## Adagrad

## Adadelta

## RMSprop

## Adam

## AdaMax

## AMSGrad

## Nadam





# Reference

http://ruder.io/optimizing-gradient-descent/
