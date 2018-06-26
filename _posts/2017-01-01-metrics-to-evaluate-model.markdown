---
title: "机器学习模型的评价方法和指标"
subtitle: "Metrics To Evaluate Machine Learning Model"
categories: [design]
layout: post
---

# Online metrics vs Offline metrics

Online metrics是直接在线上环境做AB测试，比较两个实验组的核心指标，比如CTR、转化率等。

Offline metrics是希望在模型上线之前，使用历史数据进行效果评估。离线指标有些是纯数学模型指标，有些是结合实际问题的量化指标。




# Classification Metrics

## Classification Accuracy


## Logarithmic Loss

Logarithmic loss (related to cross-entropy) measures the performance of a classification model where the prediction input is a probability value between 0 and 1. The goal of our machine learning models is to minimize this value. A perfect model would have a log loss of 0. Log loss increases as the predicted probability diverges from the actual label.


## Confusion Matrix


## AUC Area under Curve

AUC metric is only designed for binary classifiers

## Classification Report

## F1 Score






# Regression Metrics

Mean Absolute Error
Mean Squared Error
R^2 Metric



# Clustering metrics

‘adjusted_mutual_info_score’	metrics.adjusted_mutual_info_score	 
‘adjusted_rand_score’	metrics.adjusted_rand_score	 
‘completeness_score’	metrics.completeness_score	 
‘fowlkes_mallows_score’	metrics.fowlkes_mallows_score	 
‘homogeneity_score’	metrics.homogeneity_score	 
‘mutual_info_score’	metrics.mutual_info_score	 
‘normalized_mutual_info_score’	metrics.normalized_mutual_info_score	 
‘v_measure_score’	metrics.v_measure_score






# Learning to Rank Metric

如何评价排序的好坏是一个非常重要的事情。否则我们无法知道怎么去优化自己的算法。评价这件事是我们在训练时进行的。这不是在评价样本数据的好坏，也不是为了评价某一次预测的结果，而是为了把模型对所有训练数据的表现整体来看。

Information Retrieval的评价指标包括：MRR，MAP，ERR，NDCG等


## MAP Mean average precision

Mean average precision for a set of queries is the mean of the average precision scores for each query.

其中，Precision at i is a percentage of correct items among first i recommendations.



## NDCG Normalized Discounted Cumulative Gain


MRR







# References

scikit-learn中的metric http://scikit-learn.org/stable/modules/model_evaluation.html

https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)

https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/

https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234
