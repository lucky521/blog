---
title: "个性化算法"
subtitle: "Personalization & Recommendation Algorithm"
categories: [Tech]
layout: post
---

由于推荐领域对个性化算法的依赖度很高，所以推荐算法和个性化算法是比较类似的概念。

个性化算法不仅仅可以当做推荐算法用于推荐系统，还可以用于排序系统。很多时候我们可以把推荐看做是一种排序。

本文总结了常用的可用于各个场景实现个性化需求的个性化算法。


# 用户画像描述

用户画像是在互联网中对某一个具体用户的属性描述，不是标准一寸照、也不是户口页信息，而是关于这一用户的抽象身份信息，用户画像的构建源于用户的各种网络行为数据，构建成的画像最终会被用于搜索、推荐、广告等互联网业务或其他AI产品。
本文会记录一些关于用户画像构建和使用的思考，不会涉及工作业务具体的内容。

用户画像的数据始源有搜索记录、点击记录、购买记录、位置信息、账户信息。这些都是用户在互联网中不得不产生出并透漏给企业的“隐私数据”。

## 长期的用户属性

长期的用户属性反映了一个用户基本的特征，比如性别、年龄、地域、职业、学历。

## 短期的用户意图

短期的特征表现的用于近期或实时的意图，比如最近想买什么东西、最近关注什么品牌，比如现在喜欢什么款式、想要什么价位的、会重点关注哪一方面因素（关键词、评价、人气、品牌、服务、商家）。

即使是同一个人，也存在闲逛、比价、收藏、购买等不同的情况。


# 有了用户画像，用来做什么？

让用户尽快发现它最想发现的东西。根据画像来推荐，根据画像来排序。

挖掘长尾（The Long Tail）中对个人有价值的东西。

基于用户画像，互联网服务可以针对每一个账户或者设备来提供定制化的服务。这样的定制是通过数据挖掘训练、使用机器学习模型生成的方案，相比于人工定制来讲，会更加简单、快速、准确。





# 协同过滤




# Restricted Boltzmann Machines

https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf

# SVD & SVD++


# 矩阵分解

Latent-factor models

Matrix Factorization





# Contextual Bandits

simplified version of reinforcement learning

https://zhuanlan.zhihu.com/p/32441807

https://zhuanlan.zhihu.com/p/35753281


# 聚类




# 神经网络




# 知识图谱







# 学习资料

Machine Learning Summer School 2014 in Pittsburgh
http://technocalifornia.blogspot.com/2014/08/introduction-to-recommender-systems-4.html

Predicting movie ratings and recommender systems
http://arek-paterek.com/book/





# Reference

https://zhuanlan.zhihu.com/p/27768663

https://www.msra.cn/zh-cn/news/executivebylines/tech-bylines-personalized-recommendation-system

https://www.quora.com/Which-algorithms-are-used-in-recommender-systems

http://technocalifornia.blogspot.com/2014/08/introduction-to-recommender-systems-4.html

一淘的个性化搜索技术  https://zhuanlan.zhihu.com/p/22516029

淘宝的千人千面  https://www.zhihu.com/question/21353219

