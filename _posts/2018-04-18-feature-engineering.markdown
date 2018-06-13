---
title: "特征工程"
subtitle: "Feature Engineering"
categories: [design]
layout: post
---

特征 Feature 指的是机器学习模型和算法的输入数据。

算法的目的是解决真实世界的问题，而算法本身又是数学抽象的，在特征工程里就必须要把现实世界的数据抽象、转换、整理为可以被算法所利用的特征数据。再好的算法，没有给它提供适合、足够的特征，也不会是一个好模型。


# 区分特征提取 & 特征处理 & 特征选择

模型简单来说是一个 特征向量X -> Y标签值 的函数。对于X，我们一般需要做三件事：
一个是把现实世界原始的数据转换成数学抽象的值，这样才能被程序使用。
一个是对特征做数学处理，使其能更好的被算法所利用。
一个是找出确实会对Y会产生影响的X分量，抛弃对Y来讲毫无意义的特征。

这三个事儿分别叫做特征提取、特征处理、特征选择。

特征提取是指将机器学习算法不能识别的原始数据转化为算法可以识别的特征的过程。

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







# 特征处理 Feature Processing

## 标准化 Standardization

ompute its mean (u) and standard deviation (s) and do x = (x - u)/s.       
放缩到均值为0，方差为1

## 归一化 Normalization

x = (2x - max - min)/(max - min).    
线性放缩到[-1,1]

‘l1’, ‘l2’, or ‘max’

## 二值化 Binarization

定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0.


## 缺失值计算 Imputation

来自真实世界经常是不完整的，有些特征在某一行可能为空。“空”本身是不能被模型所理解的，我们需要做一定的推测，给这些缺失位置赋予一个估计的数值，比如一个平均值或中位数值。


## 对定性特征/离线特征编码

我们的算法不能把离散型特征作为输入，需要先把离散型特征编码成为连续型特征。 OneHotEncoder是最常用的一种编码方式。


## 多项式特征扩展

PolynomialFeatures 以多项式运算的方式将2个以上的特征以多项式形式组合，产生新的扩展特征。







# 特征选择 Feature Selection

## PCA 主成分分析




# 特征交叉 Feature Interaction / 特征组合 Feature Cross

组合两个特征，我们可以进行加减乘除运算。

Sum of two features: Let’s say you wish to predict revenue based on preliminary sales data. You have the features sales_blue_pens and sales_black_pens. You could sum those features if you only care about overall sales_pens.

Difference between two features: You have the features house_built_date and house_purchase_date. You can take their difference to create the feature house_age_at_purchase.

Product of two features: You’re running a pricing test, and you have the feature price and an indicator variable conversion. You can take their product to create the feature earnings.

Quotient of two features: You have a dataset of marketing campaigns with the features n_clicks and n_impressions. You can divide clicks by impressions to create  click_through_rate, allowing you to compare across campaigns of different volume.

除了组合两个特征，我们也可以组合多个特征。




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





# Reference

https://www.zhihu.com/question/29316149

http://scikit-learn.org/stable/modules/classes.html
