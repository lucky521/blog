---
title: "特征工程"
subtitle: "Feature Engineering"
categories: [MachineLearning]
layout: post
---

特征 Feature 指的是机器学习模型和算法的输入数据。

算法的目的是解决真实世界的问题，而算法本身又是数学抽象的，在特征工程里就必须要把现实世界的数据抽象、转换、整理为可以被算法所利用的特征数据。再好的算法，没有给它提供适合、足够的特征，也不会是一个好模型。


# 特征

## 区分 数据的特征 & 矩阵的特征

Eigenvalue

![](https://img-blog.csdn.net/20161018093106253?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

[参考](https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E5%80%BC%E5%92%8C%E7%89%B9%E5%BE%81%E5%90%91%E9%87%8F)

## 区分 特征提取 & 特征处理 & 特征选择

模型简单来说是一个 特征向量X -> Y标签值 的函数。对于X，我们一般需要做三件事：
一个是把现实世界原始的数据转换成数学抽象的值，这样才能被程序使用。
一个是对特征做数学处理，使其能更好的被算法所利用。
一个是找出确实会对Y会产生影响的X分量，抛弃对Y来讲毫无意义的特征。

这三个事儿分别叫做特征提取、特征处理、特征选择。

特征提取是指将机器学习算法不能识别的原始数据转化为算法可以识别的特征的过程。来自自然界的特征本身可能并不是数学形式，或者原始值虽是数值但特征值之间没有任何偏序关系。特征提取这个过程类似于编码的过程。

特征处理也可以称作特征预处理，处理的手段有很多，比如 scaling, centering, normalization, binarization, imputation。

特征选择是指去掉无关特征，保留相关特征的过程，也可以认为是从所有的特征中选择一个最好的特征子集。特征选择本质上可以认为是降维的过程。




# 从数据到特征

## 结构化数据和非结构化数据

结构化数据：比如k-v数据、表格数据。每个数据都有其意义定义、取值范围。

非结构化数据：比如图片、音频、视频、语言。其取值范围不定，意义也不能被计算机直接理解。

## 连续型特征和离散型特征

连续型特征, continuous features，又叫数值特征,是指取值本身就蕴含数值大小关系的特征。比如价格、尺寸。

离散型特征, categorical features，又叫定性特征：是指取值没有偏序关系的特征。比如无意义的编号、颜色（红黄蓝）、语言文字。

## 特征向量到特征矩阵

不管是图像像素、声音比特、汉语文字，还是表格数据，最后要交给机器学习算法来训练的“特征”总是一个矩阵，叫特征矩阵。其每一行是数据集的一条数据，叫作特征向量。其每一列是众多特征中的一个特征。








# 特征提取 Feature Extraction

Feature hashing

The Bag of Words

Feature Embedding




# 特征处理 Feature Processing

## 标准化 Standardization

0均值标准化/z-score/均值-标准差缩放

0均值归一化方法将原始数据集归一化为均值为0、方差1的数据集.

z = (x-μ)/σ.
μ、σ分别为原始数据集的均值和方差.
该种归一化方式要求原始数据的分布可以近似为高斯分布，否则归一化的效果会变得很糟糕。

在sklearn中对应的方法是preprocessing.scale


## 归一化 Normalization

线性归一化(min-max标准化)

x = (2x - max - min)/(max - min).    
线性放缩到[-1,1]


在sklearn中对应的方法是preprocessing.MinMaxScaler




‘l1’, ‘l2’, or ‘max’

## 二值化 Binarization

定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0.


## 缺失值计算 Imputation

也称作是 Matrix Completion。
来自真实世界经常是不完整的，有些特征在某一行可能为空。“空”本身是不能被模型所理解的，我们需要做一定的推测，给这些缺失位置赋予一个估计的数值，比如一个平均值或中位数值。

XGBoost会通过loss的计算过程自适应地判断特征值缺失的样本被划分为左子树还是右子树更优。

Missing value layer 美团机器学习实践中提到一种方法，用一个网络层来学习缺失值的权重。通过这一层，缺失的特征根据对应特征的分布来自适应的学习出一个合理的取值。


## 对定性特征/离线特征编码

我们的算法不能把离散型特征作为输入，需要先把离散型特征编码成为连续型特征。 OneHotEncoder是最常用的一种编码方式。

一种方法叫 Label Encoder： 将文本形式的类别特征转换为数值型。相当于给文本形式的类别编号（数字编号1,2,3,4...）。
这个方法的缺点是编号之后，默认可能会误认为这些类别之间具有偏序关系（其实并没有）。

```
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
x[:, 0] = labelencoder.fit_transform(x[:, 0])
```

另一种方法叫 One Hot Encoder：为了不加入类别编号的偏序关系，对每一个类别都加一列，是这个类别则为1，非这个类别则为0.

```
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
```

## 特征离散化

将连续性特征转换为离散特征、或者二值特征。
有些特征虽然也是数值型的，但是该特征的取值相加相减是没有实际意义的，那么该数值型特征也要看成离散特征，采用离散化的技术。

在sklearn中对应的方法是preprocessing.KBinsDiscretizer

## 多项式特征扩展

PolynomialFeatures 以多项式运算的方式将2个以上的特征以多项式形式组合，产生新的扩展特征。








# 特征选择 Feature Selection

为什么会有维度灾难（curse of dimensionality）？
特征维度越高，所需要的样本点数据越多。
聚类会变得困难。


## 分析特征值的分布

画出特征取值分布图，计算方差


## 计算特征之间的相关系数



## PCA 主成分分析

PCA降维的大致思想是挑选特征明显的、显得比较重要的信息保留下来。

当数据没有类别标签的时候，


## LDA 线性判别分析

当数据有类别标签的时候，使用LDA来最大化类间距离、最小化类内距离。



## SVD 奇异值分解 - singular value decomposition

线性代数中一种重要的矩阵分解

## Sammon’s Mapping 方法



## 特征交叉 Feature Interaction / 特征组合 Feature Cross

组合两个特征，我们可以进行加减乘除运算。

Sum of two features: Let’s say you wish o predict revenue based on preliminary sales data. You have the features sales_blue_pens and sales_black_pens. You could sum those features if you only care about overall sales_pens.

Difference between two features: You have the features house_built_date and house_purchase_date. You can take their difference to create the feature house_age_at_purchase.

Product of two features: You’re running a pricing test, and you have the feature price and an indicator variable conversion. You can take their product to create the feature earnings.

Quotient of two features: You have a dataset of marketing campaigns with the features n_clicks and n_impressions. You can divide clicks by impressions to create  click_through_rate, allowing you to compare across campaigns of different volume.

除了组合两个特征，我们也可以组合多个特征。




# 流形学习 Manifold Learning

流形学习方法是模式识别中的基本方法，分为线性流形学习算法和非线性流形学习算法。

非线性流形学习算法包括等距映射（Isomap） ，拉普拉斯特征映射（Laplacian eigenmaps，LE） ，局部线性嵌入(Locally-linear embedding，LLE) 等。
而线性方法则是对非线性方法的线性扩展，如主成分分析（Principal component analysis，PCA），多维尺度变换（Multidimensional scaling，MDS）等。


# 度量学习 Metric Learning

学习样本之间的相似程度。



# 稀疏表示 Sparse representation 和 字典学习 Dictionary learning

字典学习（Dictionary Learning）和稀疏表示（Sparse Representation）在学术界的正式称谓应该是稀疏字典学习（Sparse Dictionary Learning）。该算法理论包含两个阶段：字典构建阶段（Dictionary Generate）和利用字典（稀疏的）表示样本阶段（Sparse coding with a precomputed dictionary）。

稀疏表示的本质：用尽可能少的资源表示尽可能多的知识，这种表示还能带来一个附加的好处，即计算速度快。

字典学习总是尝试学习蕴藏在样本背后最质朴的特征（假如样本最质朴的特征就是样本最好的特征）。




# 处理数据倾斜/不平衡

类不平衡（class-imbalance）是指在训练分类器中所使用的训练集的类别分布不均。比如说一个二分类问题，1000个训练样本，比较理想的情况是正类、负类样本的数量相差不多；而如果正类样本有995个、负类样本仅5个，就意味着存在类不平衡。


解决思路：

    1. 对较多的那个类别进行欠采样(under-sampling)，舍弃一部分数据，使其与较少类别的数据相当
    2. 对较少的类别进行过采样(over-sampling)，重复使用一部分数据，使其与较多类别的数据相当
    3. 分类阈值调整（threshold moving），将原本默认为0.5的阈值调整到 较少类别/（较少类别+较多类别）即可
    4. 在损失函数中使用类权重：本质上，代表性不足的类在损失函数中获得更高的权重，因此对该特定类的任何错误分类将导致损失函数中的非常高的误差。



解决办法：

    1. 对数据进行采样的过程中通过相似性同时生成并插样“少数类别数据”，叫做SMOTE过采样算法
    2. 对数据先进行聚类，再将大的簇进行随机欠采样或者小的簇进行数据生成
    3. 把监督学习变为无监督学习，舍弃掉标签把问题转化为一个无监督问题，如异常检测
    4. 先对多数类别进行随机的欠采样，并结合boosting算法进行集成学习










# 从特征到训练集

在训练一个模型时，不仅需要特征向量，还需要每个特征向量所对应的标签Y。
对于分类问题，Y指的是类别。对于回归问题，Y指的是分值。训练集就是一个个X到Y的映射集合Set(X->Y)。

如何得到训练集呢？一方面要从原始数据中得到特征向量，一方面是要定义该特征向量所对应的标签Y。
前者是前几章所讲的事情，特征不管是单边特征、双边特征、多边特征，都要表达抽象的量化信息，而不可以表达针对某一对象的具体信息。也就是“对数不对人，对数不对物”。

后者这个目标分值一定是能直接从原始数据中得到或简单计算得到。要么需要人工打标，要么需要特定的log系统来记录训练数据。

## 定义样本的label

对于二分问题，样本的label就是0/1，称为正样本和负样本。
对于回归问题，样本的label是一个数，可以称为score分数、也可以称为概率值。

比如说对于排序或广告，我们可以把“有点击”的文档样本作为正样本，把“没有点击”的文档样本作为负样本。
或者是把“有无订单”作为这个文档样本的label。




# Candidate Sampling


Candidate Sampling 训练方法要对每一次训练任务构建训练集合，每一个集合都是总样本集的一个小子集。

Context-Specific Sampling

Generic Sampling



参考 https://www.tensorflow.org/extras/candidate_sampling.pdf











# 样本数据的存储格式

## CSV格式 和 TSV格式

其文件以纯文本形式存储表格数据（数字和文本），文件的每一行都是一个数据记录。
一般需要一个首行来表明每一列的字段名称。
列与列之间由逗号或者tab分割。

Comma-separated values。

Tab-separated values。


## svmlight格式

svmlight是一种简化的文本格式。最早源自于libsvm。


## TFRecord格式

TFRecord是Tensorflow和TFLearn所特有的二进制形式的样本文件格式。



## Feather格式

Feather数据格式是为R、Python、Julia语言可以支持的数据文件格式，


## libffm格式

libffm格式是FM实现库libffm所支持的格式。

The data format of LIBFFM is:

<label> <field1>:<feature1>:<value1> <field2>:<feature2>:<value2> ...
.
.
.

`field' and `feature' should be non-negative integers






# Reference

https://www.zhihu.com/question/29316149

http://scikit-learn.org/stable/modules/classes.html
