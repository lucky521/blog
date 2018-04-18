---
title: "特征工程"
subtitle: "Feature Engineering"
categories: [design]
layout: post
---

特征 Feature 指的是机器学习模型和算法的输入数据。

算法的目的是解决真实世界的问题，而算法本身又是数学抽象的，在特征工程里就必须要把现实世界的数据抽象、转换、整理为可以被算法所利用的特征数据。再好的算法，没有给它提供适合、足够的特征，也不会是一个好模型。

# 特征选择和特征提取

模型简单来说是一个 特征向量X -> Y标签值 的函数。对于X，我们一般需要做三件事：
一个是找出确实会对Y会产生影响的X分量，抛弃对Y来讲毫无意义的特征。
一个是把现实世界原始的数据转换成数学抽象的值，这样才能被程序使用。
一个是对特征做数学处理，使其能更好的被算法所利用。

这三个事儿分别叫做特征选取、特征提取、特征处理。

特征选择是指去掉无关特征，保留相关特征的过程，也可以认为是从所有的特征中选择一个最好的特征子集。特征选择本质上可以认为是降维的过程。

特征提取是指将机器学习算法不能识别的原始数据转化为算法可以识别的特征的过程。

特征处理也可以称作特征预处理，处理的手段有很多，比如 scaling, centering, normalization, binarization, imputation。


# 特征选择 Feature Selection



# 特征提取 Feature Extraction



# 特征处理 Feature Processing

## 标准化

## 规范化

## 二值化



# Reference

https://www.zhihu.com/question/29316149

http://scikit-learn.org/stable/modules/classes.html
