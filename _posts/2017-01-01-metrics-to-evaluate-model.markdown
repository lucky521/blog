---
title: "机器学习模型的评价方法和指标"
subtitle: "Metrics To Evaluate Machine Learning Model"
categories: [design]
layout: post
---

# Online metrics vs Offline metrics

Online metrics是直接在线上环境做AB测试，比较两个实验组的核心指标，比如CTR、转化率等。

Offline metrics是希望在模型上线之前，使用历史数据进行效果评估。离线指标有些是纯数学模型指标，有些是结合实际问题的量化指标。


# 基础统计数据

考虑一个二分问题，即将实例分成正类（positive）或负类（negative）。对一个二分问题来说，会出现四种情况。如果一个实例是正类并且也被预测成正类，即为真正类（True positive）,如果实例是负类被预测成正类，称之为假正类（False positive）。相应地，如果实例是负类被预测成负类，称之为真负类（True negative）,正类被预测成负类则为假负类（false negative）。


                  预测1                     预测0
      实际1        True Positive(TP)        False Negative(FN)
      实际0        False Positive(FP)        True Negative(TN)


TP：正确肯定的数目；
FN：漏报，没有正确找到的匹配的数目；
FP：误报，给出的匹配是不正确的；
TN：正确拒绝的非匹配对数；


## TPR、FPR & TNR

真正类率(True Positive Rate, TPR), 也称为Sensitivity，计算公式为
TPR = TP / (TP + FN).
计算的是分类器所识别出的正实例占所有正实例的比例。

负正类率(False Positive Rate, FPR),，也成为1-Specificity，计算公式为
FPR = FP / (FP + TN).
计算的是分类器错认为正类的负实例占所有负实例的比例。

真负类率（True Negative Rate，TNR），也称为specificity，计算公式为
TNR = TN /(FP + TN) = 1 - FPR



## 精确率Precision、召回率Recall

准确率 Precision = 提取出的正确信息条数 /  提取出的信息条数    

召回率 Recall = 提取出的正确信息条数 /  样本中的信息条数    



# 分类模型评价指标 Classification Metrics

## F1 Score

为了能够评价不同算法的优劣，在Precision和Recall的基础上提出了F1值的概念，来对Precision和Recall进行整体评价。F1的定义如下：
F1值  = 正确率 * 召回率 * 2 / (正确率 + 召回率)

## AUC Area under Curve

AUC metric is only designed for binary classifiers.

机器学习实践中分类器常用的评价指标就是auc，不想搞懂，简单用的话，记住一句话就行。auc取值范围[0.5, 1]，越大表示越好，小于0.5的把结果取反就行。
roc曲线下的面积就是auc，所以要先搞清楚roc。

ROC曲线的横坐标为FPR，纵坐标为TPR。ROC曲线越靠拢(0,1)点，越偏离45度对角线越好，Sensitivity、Specificity越大效果越好。

AUC(Area under Curve)：Roc曲线下的面积.

## Classification Accuracy


## Logarithmic Loss

Logarithmic loss (related to cross-entropy) measures the performance of a classification model where the prediction input is a probability value between 0 and 1. The goal of our machine learning models is to minimize this value. A perfect model would have a log loss of 0. Log loss increases as the predicted probability diverges from the actual label.


## Confusion Matrix



## Classification Report







# 回归模型评价指标 Regression Metrics

Mean Absolute Error
Mean Squared Error
R^2 Metric



# 聚类模型评价指标 Clustering metrics

‘adjusted_mutual_info_score’	metrics.adjusted_mutual_info_score	 
‘adjusted_rand_score’	metrics.adjusted_rand_score	 
‘completeness_score’	metrics.completeness_score	 
‘fowlkes_mallows_score’	metrics.fowlkes_mallows_score	 
‘homogeneity_score’	metrics.homogeneity_score	 
‘mutual_info_score’	metrics.mutual_info_score	 
‘normalized_mutual_info_score’	metrics.normalized_mutual_info_score	 
‘v_measure_score’	metrics.v_measure_score






# 排序模型评价指标 Learning to Rank Metric

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
