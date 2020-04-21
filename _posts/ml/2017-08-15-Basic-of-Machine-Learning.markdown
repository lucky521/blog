---
title: "机器学习的基础理论"
categories: [MachineLearning]
layout: post
---

# 机器学习领域的基础理论书籍推荐

    Machine Learning:A Probabilistic Perspective

    Pattern Recognition and Machine Learning

    The Elements of Statistical Learning

    Elements of Information Theory

    An Introduction to Statistical Learning with Application in R

    An Introduction to Computational Learning Theory


# Probability & Statistics 概率vs统计

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









# 频率学派 vs 贝叶斯学派

这两者是统计学中的两大学派。

贝叶斯学派认为不存在真正意义的随机。我们所谓的随机，其实是对未知的估计，其中的发生概率是一个主观经验值。而频率学派认为随机是客观的随机，以客观的概率发生。

最大似然估计是频率学派的学习方法；最大后验估计是贝叶斯学派的学习方法。

## 最大似然估计 Maximum Likelihood Estimation
似然函数是指样本视为不变量，分布参数视为变量时的分布函数。
最大似然估计会寻找关于 分布参数 的最可能的值（即，在所有可能的取值中，寻找一个值使当前样本的“可能性”最大化）。

最大似然估计提供了一种给定观察数据来评估模型参数的方法，即：“模型已定，参数未知”。

比如我假设一个模型满足正态分布，我通过采样，获取部分样本，然后通过最大似然估计来获取上述假设中的正态分布的均值与方差。

最大似然估计中采样需满足一个很重要的假设，就是所有的采样都是独立同分布的。


## 最大后验估计 Maximum A Posteriori estimation
MAP优化的是一个后验概率，样本视为不变量时，分布参数视为变量，寻找能使得后验概率值最大的分布参数。

最大似然估计是求参数θ, 使似然函数P(x0|θ)最大。
最大后验概率估计则是想求θ使P(x0|θ)P(θ)最大。求得的θ不单单让似然函数大，θ自己出现的先验概率也得大。


## 比较两种方法(MLE vs MAP)
MLE只考虑样本集，样本集完全决定了参数估计结果。
而贝叶斯学习通过先验分布，可以将一定的先验知识”编码“进去，从而在某些现实中更贴合。


## Frequentist Statistics

?


## 贝叶斯统计 Bayesian Statistics

三种信息：总体分布、样本分布、先验分布。

贝叶斯统计的特点是使用先验信息，

把未知量当做随机变量。


## 贝叶斯估计 Bayesian Inference
MLE is a special case of MAP where the prior is uniform.





# Statistical learning theory 统计学习理论 vs Computational learning theory 计算学习理论

[两个概念的差异和关联是什么？](https://www.quora.com/Whats-the-difference-between-statistical-learning-theory-and-computational-learning-theory)

- 统计学习理论（LST）：研究一个模型的错误边界（error bound,  bound between generalization error and training error）。
- 计算学习理论(CoLT)：研究一个问题是不是可学习的。

## 统计学习理论
统计学习本质上是一个最优化问题。优化的是映射函数f(x->y)，使得损失函数L(x,y,f)的期望最小。这里的f是未知的，L是要我们来定义。


## 计算学习理论
分析学习任务的困难本质，为学习算法提供理论保证，并根据分析结果指导算法设计。






# 生成方法 vs 判别方法

判别模型求解的思路是：学习条件分布------>模型参数后验概率最大------->（似然函数\cdot 参数先验）最大------->最大似然。

生成模型的求解思路是：学习联合分布------->求解类别先验概率和类别条件概率






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






# PAC Learning - Probably Approximately Correct 

PAC-learning在1984年由Leslie Valiant提出，由这个概念产生了计算学习理论。

[Probably Approximately Correct — a Formal Theory of Learning](https://jeremykun.com/2014/01/02/probably-approximately-correct-a-formal-theory-of-learning/)


什么叫做PAC-learning？

    A problem is PAC-learnable if there is an algorithm A which for any distribution D and any concept/target c will, when given some independently drawn samples and with high probability, produce a hypothesis whose error is small.



# 假设空间、版本空间


# 最小描述长度学习


# Vapnik-Chervonenkis Dimension VC维

VC理论研究的是一种复杂度的衡量方法。

VC维是一个很有趣的概念，它的主体是一类函数，描述的是这类函数能够把多少个样本的所有组合都划分开来。VC维的意义在哪里呢？它在于当你选定了一个模型以及它对应的特征之后，你是大概可以知道这组模型和特征的选择能够对多大的数据集进行分类的。此外，一类函数的VC维的大小，还可以反应出这类函数过拟合的可能性。







# 误差 Error

借助”有效”假设函数和 VC 维的概念来推导泛化误差上界。


残差：目标值和预估值的差.





# 正则化 和 bias-variance tradeoff

bias-variance tradeoff可以被翻译做”偏差-方差困境“。

如果说现阶段我国的主要矛盾是“人民日益增长的美好生活需要和不平衡不充分的发展之间的矛盾”，那么机器学习中的主要矛盾就是模型要尽量拟合数据和模型不能过度拟合数据之间的矛盾。而化解这一矛盾的核心技术之一就是正则化。正则化的具体方法不在此讨论，但需要理解的，是各种正则化方法背后透露出的思想：bias-variance tradoff。在不同利益点之间的平衡与取舍是各种算法之间的重要差异，理解这一点对于理解不同算法之间的核心差异有着非常重要的作用。







# 信息论

从某种角度来讲，机器学习和信息论是同一个问题的两个侧面，机器学习模型的优化过程同时也可以看作是最小化数据集中信息量的过程。对信息论中基本概念的了解，对于机器学习理论的学习是大有裨益的。例如决策树中用来做分裂决策依据的信息增益，衡量数据信息量的信息熵等等，这些概念的理解对于机器学习问题神本的理解都很有帮助。





# 最优化理论

绝大多数机器学习问题的解决，都可以划分为两个阶段：建模和优化。所谓建模就是后面我们会提到的各种用模型来描述问题的方法，而优化就是建模完成之后求得模型的最优参数的过程。机器学习中常用的模型有很多，但背后用到的优化方法却并没有那么多。换句话说，很多模型都是用的同一套优化方法，而同一个优化方法也可以用来优化很多不同模型。对各种常用优化方法的和思想有所有了解非常有必要，对于理解模型训练的过程，以及解释各种情况下模型训练的效果都很有帮助。这里面包括最大似然、最大后验、梯度下降、拟牛顿法、L-BFGS等。






# 概率图模型

Probabilistic Graphical Model 概率图模型 是用图论方法以表现若干个独立随机变量之关联的一种建模法。
概率图模型有朴素贝叶斯模型、贝叶斯网络、最大熵模型、隐马尔可夫模型HMM、条件随机场CRF、主题模型.







# 常用的数学函数

softplus

ReLU

sigmoid

softmax

## Softmax-based Approaches

Hierarchical Softmax

Differentiated Softmax

## Cross Product Transformation

https://datascience.stackexchange.com/questions/57435/how-is-the-cross-product-transformation-defined-for-binary-features




# 不同机器学习模型之间的共性和联系

每一个公式、函数、参数被引入到模型中，都有其业务目的。






# 参考资料

http://blog.codinglabs.org/articles/statistical-learning-theory.html

https://www.zhihu.com/question/20446337


- Algebra, Topology, Differential Calculus, and Optimization Theory For Computer Science and Machine Learning 这本书1900页，涵盖了计算机科学所需的线性代数、微分和最优化理论等问题

- 斯坦福 CS229: Machine Learning http://cs229.stanford.edu/