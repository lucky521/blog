---
title: "使用机器学习做图像识别"
subtitle: "图像识别方法与应用"
categories: [MachineLearning]
layout: post
---

图像分类是图像检测、图像分割、物体跟踪、行为分析等其他高层视觉任务的基础。

图像分类、图像识别本质上讲，是通过大量带标记数据，来创建模型（带特征的模型或者抽象数学模型），训练好模型之后，给一个新图片，就可以对其进行分类。

和其他数据不同，图像的像素本身是没有直接意义的数据。因此一般要先通过手工特征或特征学习方法对整个图像进行全部描述，然后才使用分类器判别物体类别。

一个好的模型既要对不同类别识别正确，同时也应该能够对不同视角、光照、背景、变形或部分遮挡的图像正确识别。

# 图像分类的方法

## 模式识别分类方法(PR)

在机器学习盛行之前，图像分类主要依靠的是模式识别方法。


## 传统机器学习分类方法(ML)

传统机器学习是利用特征工程(feature engineering)，人为对数据进行提炼清洗。

传统的机器学习方法一般要经历特征提取、特征编码、特征分类三个步骤。对于图像来讲，特征（Feature）指的是能够表达其视觉属性的向量。对于机器学习算法来讲，特征（Feature）指的是算法的输入。
首先，通常从图像中按照固定步长、尺度提取大量局部特征描述。
底层特征中包含了大量冗余与噪声，为了提高特征表达的鲁棒性，需要使用一种特征变换算法对底层特征进行编码。
特征编码之后一般会经过空间特征约束。
经过前面的处理之后，一张图片就会用一个向量来描述，可以直接对其使用分类算法。

步骤：
    1 图像特征提取：
        SIFT(Scale-Invariant Feature Transform, 尺度不变特征转换)
        HOG(Histogram of Oriented Gradient, 方向梯度直方图)
        LBP(Local Bianray Pattern, 局部二值模式)
    2 特征编码：
        向量量化编码
        稀疏编码
        局部线性约束编码
        Fisher向量编码
    3 空间特征约束：
        特征汇聚是指在一个空间范围内，对每一维特征取最大值或者平均值，可以获得一定特征不变形的特征表达。
    4 分类器：
        表现比较好的分类器如使用核方法的SVM

## 深度学习分类方法(DL)

深度学习是利用表示学习(representation learning)，机器学习模型自身对数据进行提炼。

卷积神经网络模型中包含了对图像的特征提取、重采样等处理过程，因此将传统机器学习方法的几个步骤合为一体。

    卷积层(convolution layer)：
        执行卷积操作提取底层到高层的特征，发掘出图片局部关联性质和空间不变性质。
    池化层(pooling layer):
        执行降采样操作。通过取卷积输出特征图中局部区块的最大值(max-pooling)或者均值(avg-pooling)。降采样也是图像处理中常见的一种操作，可以过滤掉一些不重要的高频信息。
    全连接层(fully-connected layer）:
        输入层到隐藏层的神经元是全部连接的。
    非线性变化层:
        卷积层、全连接层后面一般都会接非线性变化层，例如Sigmoid、Tanh、ReLu等来增强网络的表达能力，在CNN里最常使用的为ReLu激活函数。
    Dropout层:
        在模型训练阶段随机让一些隐层节点权重不工作，提高网络的泛化能力，一定程度上防止过拟合。






# CNN卷积神经网络的一般结构

卷积神经网络近几年在图像视觉领域的发展使得深度学习成为大热门。

卷积神经网络是用到了多个一样的神经元，可以表达更大的模型。

我们在写程序的时候，会写一个函数，然后在多个地方调用这个函数。这是函数的复用作用。类似的，卷积神经网络学习一个神经元，然后在多个地方复用它。


假设你有一个神经网络，训练声音样本，预测是否有人声在里面。

最简单的方法是把所有声音样本按时间平分，作为同一层的输入。复杂的方法是加入数据的其他属性，比如频率。

## 卷积操作

所谓卷积神经网络，指的是至少在神经网络的某一层使用卷积运算来替代一般的矩阵乘法运算。

s(t) = (x * w)(t)

比如上面是用函数w对函数x做卷积，原函数w是t的函数x(t)，函数w称作核函数，卷积时候的新函数是s(t)。输入t可以是多维的，比如多维数组(i,j,k,...)。对于二维的情况，卷积对应了一个双重分块循环矩阵。

为什么卷积运算有助于改进神经网络？

    稀疏交互：通过核函数的大小来控制输入影响的范围。
    参数共享：非卷积的运算中参数只会用到一次。而卷积的核参数是多次使用的。
    等变表示：如果原函数是输入的平移函数，那么卷积函数对原函数有等变性。


## 池化操作

池化操作是几乎所有卷积网络都会有的操作。池化函数使用某一位置的相邻输出的总体统计特征来代替网络在该位置的输出。例如，MAX pooling函数是给出相邻矩阵区域内的最大值，当然还有平均值、加权平均值等其他池化函数。

池化的平移不变性指的是当我们对输入进行少量平移时，经过池化函数后的大多数输出不会发生变化。当我们关心一个特征是否出现而不是关心在哪里出现时，局部平移不变性是很有用的性质。





典型的卷积神经网络层有三级。

    第一级卷积层：仿射变换
    第二级探测层：整流线性 ReLU
    第三级池化层：不变性


## 卷积层 Convolutional Layer

## 全连接层 Fully-Connected Layer

## 扁平层 Flatten layer

Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。

The Flatten layer is a utility layer that flattens an input of shape n * c * h * w to a simple vector output of shape n * (c*h*w).




# CNN benchmarks

https://github.com/jcjohnson/cnn-benchmarks





# 典型的卷积神经网络模型

## LeNet

http://yann.lecun.com/exdb/lenet/

## AlexNet

从2012年这篇CNN的大作之后，卷积神经网络彻底火起来。

http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

## VGGNet

## GoogLeNet,  InceptionV1-V4

http://arxiv.org/abs/1409.4842

## ResNet

残差网络

## CaffeNet

## MobileNet

https://arxiv.org/abs/1704.04861

https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html


## DenseNet









# 典型的应用

## 文字识别

OCR是很早就有的技术。


## 图片物体标注


## 视频物体检测


## 人脸识别

在CNN尚未火热的年代，人们使用haar/lbp + adaboost级连的组合方式检测人脸，hog+svm的组合方式检测行人。其中，haar，lbp，hog等手工设计的特征提取算子用于提取特征，adaboost，svm用于对提取的特征分类。


### Openface

https://github.com/cmusatyalab/openface

openface识别人脸的工作流程：

1. 使用dlib或opencv中的现有模型识别出图像中的人脸。
2. 用神经网络对人脸像素进行调整，以使得眼镜和嘴唇锁定在固定位置。
3. 使用深度神经网络以128-dimensional unit hypersphere来表示一张脸。这种表示方法使得聚类和分类都变得容易。
4. 应用聚类或分类方法来完成识别任务。

在docker镜像里体验openface：
    http://cmusatyalab.github.io/openface/setup/
    docker pull bamos/openface
    docker run -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash

### face_recognition

https://github.com/ageitgey/face_recognition
