---
layout: post
title:  "淘宝搜索推荐论文赏析"
subtitle: ""
categories: [MachineLearning]
---

最近在arxiv下载了几篇阿里巴巴淘宝发表的，关于商品搜索排序的论文。
这里通过学习论文内容来分析一下淘宝的搜索排序和推荐实践。



## Perceive Your Users in Depth: Learning Universal User Representations from Multiple E-commerce Tasks

用户画像的通用表示学习


## Virtual-Taobao: Virtualizing Real-world Online Retail Environment for Reinforcement Learning

淘宝搜索的增强学习


## Reinforcement Learning to Rank in E-Commerce Search Engine: Formalization, Analysis, and Application

使用增强学习模型来优化排序策略

定义 search session Markov decision process 来形式化搜索过程。然后使用梯度算法来优化决策过程中的排序策略。


## A Brand-level Ranking System with the Customized Attention-GRU Model

Attention-GRU 品牌排序模型。要解决的需求是预测用户对品牌的偏好程度。

LSTM and GRU 是两种RNN模型，都具有避免梯度弥散的性质。两者相比之下，GRU参数较少，训练过程较快。我们在模型中引入了 attention 机制。



## Multi-Level Deep Cascade Trees for Conversion Rate Prediction

提出一个瀑布结构的决策树集成学习模型


## Deep Interest Network for Click-Through Rate Prediction

构建用户兴趣网络模型来预测商品的点击率
