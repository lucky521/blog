---
title: "机器学习的最优化方法"
subtitle: "Optimization Method"
categories: [MachineLearning]
layout: post
---

机器学习里模型训练的过程是在反复优化一个函数，使得其能做出理想的预测。而优化时又会设计一个Objective Function，用来衡量模型预测出来的值 h(θ) 与真实值 y 之间的差异。当然，差异越小且越稳定越好。我们所说的最优化方法，就是要优化这个Objective Function，使其取得一个极小值。

# 目标函数和目标模型

在建模的时候，这里需要考虑几个问题。

## 如何选择Objective Function？


## 如何确定满足最小化Objective Function就能训练出好的模型？


## 如何保证和确定当前的Objective Function是可优化的？


## 如何确定当前的Objective Function已取得极小值？





# 目标函数的设定 Objective Function

其实目标函数和损失函数的概念比较接近，我的理解是，目标函数=损失函数+正则化方法。
预测模型g在训练数据上的平均损失函数，我们称作是经验风险。
直接把损失函数作为目标函数的方法叫做”经验风险最小化“。
结合损失函数和正则化方法的目标函数方法叫做”结果风险最小化“。

总体来讲目标函数和损失函数都是优化的目标，我们都是要再迭代中减小其数值。

目标函数的设定中要包含 loss function 和 regularization。
前者是要使得我们的模型尽可能的学习训练样本的分布。后者是希望模型不会过拟合。

Training Loss measures how well model fit on training data.

常见的损失误差有五种：
1. 铰链损失（Hinge Loss）：主要用于支持向量机（SVM） 分类中；
2. 互熵损失 (Cross Entropy Loss，Softmax Loss)：用于Logistic 回归与Softmax 分类中；
3. 平方损失（Square Loss）：主要是最小二乘法（OLS）中；
4. 指数损失（Exponential Loss） ：主要用于Adaboost 集成学习算法中；
5. 其他损失（如0-1损失，绝对值损失）

Regularization measures complexity of model.

常见的正则化方法有三种：
1. L0值: 原向量中非0元素的个数； 
2. L1值：原向量中各元素绝对值之和；
3. L2值：原向量中个元素的平方和的平方根。




# 损失函数 Loss functions

按任务的目标我们将损失函数分为回归损失函数、分类损失函数、Embedding损失函数。

## Regression loss functions

They are used in case of regressive problems, that is when the target variable is continuous.

0. Mean Square Error (MSE)
均方误差/平方损失/L2 损失
Loss2 = (pred - actual)^2
```
def MSE(yHat, y):
    return np.sum((yHat - y)**2) / y.size
```

Most widely used regressive loss function is Mean Square Error. Other loss functions are:
1. Absolute error
measures the mean absolute value of the element-wise difference between input;

2. Smooth Absolute Error 
a smooth version of Abs Criterion.

3. 平均绝对误差(Mean Absolute Error) (MAE) /L1 损失
Loss1 = abs(pred - actual)
```
def L1(yHat, y):
    return np.sum(np.absolute(yHat - y))
```

4. 平均偏差误差（mean bias error） (MBE)
```
MBE=1n∑i=1n(y~i−yi)
```

5. Huber Loss
Loss = delta^2 * (sqrt(1 + ((pred - actual)/delta)^2) - 1)
```
def Huber(yHat, y, delta=1.):
    return np.where(np.abs(y-yHat) < delta,.5*(y-yHat)**2 , delta*(np.abs(y-yHat)-0.5*delta))
```


## Classification loss functions

The output variable in classification problem is usually a probability value f(x), called the score for the input x. Generally, the magnitude of the score represents the confidence of our prediction. The target variable y, is a binary variable, 1 for true and -1 for false.
On an example (x,y), the margin is defined as yf(x). The margin is a measure of how correct we are. Most classification losses mainly aim to maximize the margin. Some classification algorithms are:

1. Binary Cross Entropy

2. Negative Log Likelihood

3. Margin Classifier

4. Soft Margin Classifier

5. Hinge Loss/多分类 SVM 损失
Loss = max(0, 1 - (pred * actual))

```python
def Hinge(yHat, y):
    return np.max(0, 1 - yHat * y)
```

6. 交叉熵损失/负对数似然/Logarithmic Loss
Cross-entropy loss, 或者叫做 Log loss
Loss = -actual * (log(pred)) - (1-actual)(log(1-pred))

```python
def CrossEntropy(yHat, y):
    if y == 1:
      return -log(yHat)
    else:
      return -log(1 - yHat)
```

7. Sigmoid entropy loss
Loss = -actual * (log(sigmoid(pred))) - (1-actual)(log(1-sigmoid(pred)))
或者
Loss = max(actual, 0) - actual * pred + log(1 + exp(-abs(actual)))

```python
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(labels=x_vals, logits=targets)
xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)
```

8. Weighted (softmax) cross entropy loss
Loss = -actual * (log(pred)) * weights - (1-actual)(log(1-pred))
或者
Loss = (1 - pred) * actual + (1 + (weights - 1) * pred) * log(1 + exp(-actual))

```python
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)
```


9. KL散度 KL-divergence Loss
我们可以把最大似然看作是使模型分布尽可能地和经验分布相匹配的尝试。最小化训练集上的经验分布和模型分布之间的差异。
```python
def kl_divergence(p, q):  # q,p都是长度相同的浮点数向量，且向量元素值之和都为1
    return tf.reduce_sum(p * tf.log(p/q)) 
```

10. Softmax entropy loss
Loss = -actual * (log(softmax(pred))) - (1-actual)(log(1-softmax(pred)))
```python
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(unscaled_logits, target_dist)
```

11. Sparse entropy loss
Loss = sum( -actual * log(pred) )
```
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(unscaled_logits, sparse_target_dist)
```

### logits是什么？

The vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function. If the model is solving a multi-class classification problem, logits typically become an input to the softmax function. The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.


## Embedding loss functions

It deals with problems where we have to measure whether two inputs are similar or dissimilar. Some examples are:
1. L1 Hinge Error - Calculates the L1 distance between two inputs.

2. Cosine Error - Cosine distance between two inputs.

3. Noise Contrastive Estimation training loss (NCE loss)
为什么NCE常作为word2vec的loss函数？
word2vec用二分类目标来区分真实目标和噪音目标。训练的目标就是增大真实目标的输出结果，减小噪音目标的输出结果。
num_sampled refers to the number of negative sampling in NCE algorithm.

https://stackoverflow.com/questions/41475180/understanding-tf-nn-nce-loss-in-tensorflow








# 正则化方法 Regularization Functions

0. L0 Regularization

L0范数是指向量中非0的元素的个数。

1. L1 Regularization

L1范数是指向量中各个元素绝对值之和。也叫做“稀疏规则算子”（Lasso regularization）。

2. L2 Regularization

 L2范数是指向量各元素的平方和然后求平方根。也叫做"Ridge regression"。

- 为什么 l1 相比于 l2 容易获得稀疏解？
https://www.zhihu.com/question/37096933

https://maristie.com/blog/differences-between-normalization-standardization-and-regularization/

3. 在L1和L2正则的基础上，人们引入L21正则(group lasso)和L22正则(exclusive sparsity)

- 什么是稀疏解？










# 约束优化问题

研究优化的学科，便是运筹学。

机器学习模型往往找不到解析解。只能通过优化算法有限次迭代模型参数，来尽可能降低损失函数的值，也就是找数值解。

近年来最流行的求数值解的优化方法就是小批量随机梯度下降算法。

## 无约束的优化问题

无约束最优化问题是优化领域研究的基础性问题。

常见的求解方法有梯度下降法、最速下降法、牛顿法、共轭梯度法（DFP法）和变尺度法（变度量法）。

对于特殊的最小二乘问题，即把均方误差作为最小化目标，有最小二乘法。

## 有等式约束的优化问题

使用拉格朗日乘子法（Lagrange Multiplier) 求解。

常利用拉格朗日对偶性将原始问题转换为对偶问题，通过解对偶问题而得到原始问题的解。

## 有不等式约束的优化问题

常使用KKT条件转换为无约束问题来求解。

[参考](https://zhuanlan.zhihu.com/p/26514613)



## 最大似然估计 和 最大期望估计




## 目标函数的凸性和光滑性

目标函数的凸性会给优化带来很大方便，原因是凸函数的任何一个局部极小点都是全局最优解。

目标函数的光滑性意味着自变量的微小变化只会引起函数值的微小变化。

目前绝大多数优化理论都是针对凸函数的。




## 一阶确定性算法

用到了目标函数在每一个状态的一阶导数信息。


## 二阶确定性算法


## 一阶随机优化算法


## 二阶随机优化算法


## 原始方法 vs 对偶方法

对偶方法是通过对偶变换，先把原始问题转换为对偶问题，针对对偶变量进行优化。



# 最优化方法

最优化方法在这里指的就是针对损失函数的优化方法。

一个有效的优化算法会随着迭代的进行使输出的模型w越来越接近于最优模型W。

常见的有三种梯度下降方法，主要区别是在使用多少数据来计算objective function的梯度。

## Batch gradient descent

计算整个数据集的渐变, 并在每次迭代中只执行一个更新。

θ=θ−η⋅∇θJ(θ)

```
for i in range(nb_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad
```

## Stochastic gradient descent

仅使用单个训练示例来计算渐变和更新参数。

θ=θ−η⋅∇θJ(θ;x(i);y(i))

```cpp
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
```

## Mini-batch gradient descent

将数据集拆分为小批, 并对每个 mini-batch 执行参数更新。

θ=θ−η⋅∇θJ(θ;x(i:i+n);y(i:i+n))

```cpp
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
```

## 牛顿方法

用目标函数的二阶泰勒展开近似该目标函数，通过求解这个二次函数的极小值来求解凸优化的搜索方向。

https://zhuanlan.zhihu.com/p/33544363

## Momentum

momentum是模拟物理里动量的概念，积累之前的动量来替代真正的梯度

## Nesterov accelerated gradient

https://zhuanlan.zhihu.com/p/22810533

## Adagrad

Adagrad其实是对学习率进行了一个约束

## Adadelta

自动调整学习率

## RMSprop

RMSprop可以算作Adadelta的一个特例：

## Adam

Adaptive Moment Estimation (Adam) combines ideas from both RMSProp and Momentum.

## AdaMax

## AMSGrad

## Nadam

## Conjugate Gradients

## BFGS

## L-BFGS

https://blog.csdn.net/dadouyawp/article/details/44179599



## 多种优化方法的结合使用

比如通过切换从Adam到SGD。
Adam在一开始表现更好，而SGD最终达到更好的全局最小值。









# Reference

超级全面的总结： http://ruder.io/optimizing-gradient-descent/

中文版： https://zhuanlan.zhihu.com/p/22252270

区分normalization/standardization/regularization：https://maristie.com/blog/differences-between-normalization-standardization-and-regularization/

