---
title: "机器学习的基础理论"
categories: [MachineLearning]
layout: post
---

机器学习领域的基础理论书籍推荐一下：

    Machine Learning:A Probabilistic Perspective

    Pattern Recognition and Machine Learning

    The Elements of Statistical Learning

    Elements of Information Theory

    An Introduction to Statistical Learning with Application in R

    An Introduction to Computational Learning Theory


# probability & statistics 概率vs统计

Induction vs Deduction

    statistics： 我们观察到一些observations，我们的目标是找到形成这些observations的underlying process。

    probability：已知underlying process，其中有一些 random variables。


PROBABILITY

    General 一般 ---> Specific 特殊

    Population 总体 ---> Sample 样本

    Model 模型 ---> Data 数据

STATISTICS

    General <--- Specific

    Population <--- Sample

    Model <--- Data



probabilistic method or model:




# 贝叶斯估计

## Maximum likelihood estimation
似然函数是指样本视为不变量，分布参数视为变量时的分布函数。
最大似然估计会寻找关于 分布参数 的最可能的值（即，在所有可能的取值中，寻找一个值使当前样本的“可能性”最大化）。

## Maximum a posteriori estimation
MAP优化的是一个后验概率，样本视为不变量时，分布参数视为变量，寻找能使得后验概率值最大的分布参数。


## 生成方法和判别方法

判别模型求解的思路是：条件分布------>模型参数后验概率最大------->（似然函数\cdot 参数先验）最大------->最大似然。

生成模型的求解思路是：联合分布------->求解类别先验概率和类别条件概率



# 概率密度估计

## 参数估计

点估计

    矩估计

    最大似然估计

    最小方差无偏估计

    贝叶斯估计

区间估计

## 非参数估计

直方图

核密度估计

k邻近估计



# Probably Approximately Correct PAC-Learning

PAC-learning在1984年由Leslie Valiant提出，由这个概念产生了计算学习理论。

[Probably Approximately Correct — a Formal Theory of Learning](https://jeremykun.com/2014/01/02/probably-approximately-correct-a-formal-theory-of-learning/)


什么叫做PAC-learning？

    A problem is PAC-learnable if there is an algorithm A which for any distribution D and any concept/target c will, when given some independently drawn samples and with high probability, produce a hypothesis whose error is small.



# 假设空间、版本空间



# Vapnik-Chervonenkis Dimension VC维

VC理论研究的是一种复杂度的衡量方法。

VC维是一个很有趣的概念，它的主体是一类函数，描述的是这类函数能够把多少个样本的所有组合都划分开来。VC维的意义在哪里呢？它在于当你选定了一个模型以及它对应的特征之后，你是大概可以知道这组模型和特征的选择能够对多大的数据集进行分类的。此外，一类函数的VC维的大小，还可以反应出这类函数过拟合的可能性。

# 信息论

从某种角度来讲，机器学习和信息论是同一个问题的两个侧面，机器学习模型的优化过程同时也可以看作是最小化数据集中信息量的过程。对信息论中基本概念的了解，对于机器学习理论的学习是大有裨益的。例如决策树中用来做分裂决策依据的信息增益，衡量数据信息量的信息熵等等，这些概念的理解对于机器学习问题神本的理解都很有帮助。

# 正则化和bias-variance tradeoff

如果说现阶段我国的主要矛盾是“人民日益增长的美好生活需要和不平衡不充分的发展之间的矛盾”，那么机器学习中的主要矛盾就是模型要尽量拟合数据和模型不能过度拟合数据之间的矛盾。而化解这一矛盾的核心技术之一就是正则化。正则化的具体方法不在此讨论，但需要理解的，是各种正则化方法背后透露出的思想：bias-variance tradoff。在不同利益点之间的平衡与取舍是各种算法之间的重要差异，理解这一点对于理解不同算法之间的核心差异有着非常重要的作用。

# 最优化理论

绝大多数机器学习问题的解决，都可以划分为两个阶段：建模和优化。所谓建模就是后面我们会提到的各种用模型来描述问题的方法，而优化就是建模完成之后求得模型的最优参数的过程。机器学习中常用的模型有很多，但背后用到的优化方法却并没有那么多。换句话说，很多模型都是用的同一套优化方法，而同一个优化方法也可以用来优化很多不同模型。对各种常用优化方法的和思想有所有了解非常有必要，对于理解模型训练的过程，以及解释各种情况下模型训练的效果都很有帮助。这里面包括最大似然、最大后验、梯度下降、拟牛顿法、L-BFGS等。





# 不同机器学习模型之间的共性和联系
