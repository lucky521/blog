---
layout: post
title:  "淘宝京东搜索推荐论文赏析"
subtitle: ""
categories: [MachineLearning]
---

最近在arxiv下载了几篇阿里巴巴淘宝、京东发表的，关于商品搜索排序的论文。
这里通过学习论文内容来分析一下淘宝的搜索排序和推荐实践。

# 淘宝

## Perceive Your Users in Depth: Learning Universal User Representations from Multiple E-commerce Tasks

用户画像的通用表示学习


## Virtual-Taobao: Virtualizing Real-world Online Retail Environment for Reinforcement Learning

淘宝搜索的增强学习

构建模拟器，让算法从买家的历史行为中学习，规划最佳商品搜索显示策略。

GAN-SD（GAN-for-Simulating-Distribution）算法模仿虚拟买家的操作和搜索请求。

MAIL方法（Multi-agent Adversarial Imitation Learning）同时学习买家规则和平台的规则，训练买家和平台产生更加真实的交互。


## Reinforcement Learning to Rank in E-Commerce Search Engine: Formalization, Analysis, and Application

使用增强学习模型来优化排序策略

定义 search session 的 Markov decision process 来形式化搜索过程。然后使用梯度算法来优化决策过程中的排序策略。

引入状态的概念，用马尔可夫决策过程对商品搜索排序问题进行建模，并尝试用深度强化学习的方法来对搜索引擎的排序策略进行实时调控。

把搜索引擎看作智能体（Agent）、把用户看做环境（Environment），则商品的搜索问题可以被视为典型的顺序决策问题。Agent每一次排序策略的选择可以看成一次试错（Trial-and-Error），把用户的反馈，点击成交等作为从环境获得的奖赏。在这种反复不断地试错过程中，Agent将逐步学习到最优的排序策略，最大化累计奖赏。而这种在与环境交互的过程中进行试错的学习，正是强化学习（Reinforcement Learning，RL）的根本思想。


## A Brand-level Ranking System with the Customized Attention-GRU Model

Attention-GRU 品牌排序模型。要解决的需求是预测用户对品牌的偏好程度。

LSTM and GRU 是两种RNN模型，都具有避免梯度弥散的性质。两者相比之下，GRU参数较少，训练过程较快。我们在模型中引入了 attention 机制。



## Multi-Level Deep Cascade Trees for Conversion Rate Prediction

提出一个瀑布结构的决策树集成学习模型


## Deep Interest Network for Click-Through Rate Prediction

构建用户兴趣网络模型来预测商品的点击率



# 京东

## Deep Reinforcement Learning for List-wise Recommendations

将推荐的过程看做一个Markov Decision process




# 参考链接

https://medium.com/@yaoyaowd/%E9%98%BF%E9%87%8C%E5%92%8C%E4%BA%AC%E4%B8%9C%E7%9A%84%E5%A2%9E%E5%BC%BA%E5%AD%A6%E4%B9%A0-%E8%83%BD%E6%AF%94learning-to-rank%E5%A5%BD%E4%B8%80%E5%80%8D-a779e3a4cd65

https://yq.aliyun.com/articles/108481
