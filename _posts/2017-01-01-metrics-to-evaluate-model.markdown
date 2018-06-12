---
title: "机器学习模型的评价方法和指标"
subtitle: "Metrics To Evaluate Machine Learning Model"
categories: [design]
layout: post
---

# Online metrics vs Offline metrics

Online metrics是直接在线上环境做AB测试，比较两个实验组的核心指标，比如CTR、转化率等。

Offline metrics是希望在模型上线之前，使用历史数据进行效果评估。


# Classification Metrics

Classification Accuracy
Logarithmic Loss
Confusion Matrix
Area under Curve
Classification Report
F1 Score




# Regression Metrics


Mean Absolute Error
Mean Squared Error
R^2 Metric



# Learning to Rank Metric

如何评价排序的好坏是一个非常重要的事情。否则我们无法知道怎么去优化自己的算法。评价这件事是我们在训练时进行的。这不是在评价样本数据的好坏，也不是为了评价某一次预测的结果，而是为了把模型对所有训练数据的表现整体来看。

Information Retrieval的评价指标包括：MRR，MAP，ERR，NDCG等

AUC Area under the curve
MAP Mean average precision
NDCG Normalized Discounted Cumulative Gain
MRR



# References

https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)

https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/

https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234
