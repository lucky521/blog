---
title: "机器学习模型的评价方法和指标"
subtitle: "Metrics To Evaluate Machine Learning Model"
categories: [MachineLearning]
layout: post
---

模型评价是机器学习最优化方法的对立面，去评价其效果。

# 选择怎样的评估指标

## Online metrics vs Offline metrics

Online metrics是直接在线上环境做AB测试，比较两个实验组的核心指标，比如CTR、转化率等。

Offline metrics是希望在模型上线之前，使用历史数据进行效果评估。离线指标有些是纯数学模型指标，有些是结合实际问题的量化指标。


## 单值评估指标 vs 多值评估指标

单值评估指标清晰明了，有利于最终的评估。如果有多个指标都是很重要的，可以将这多个值合并为一个值来表示。
比如取平均值或者加权平均值是将多个指标合并为一个指标的最常用方法之一。


## 指标的作用

一方面是让我们对当前的模型的好坏有一个量化的认知。
另一方面是在训练过程中以某一个指标作为训练算法的目标，通过优化目标来训练模型。





# 基础统计数据

考虑一个二分问题，即将实例分成正类（positive）或负类（negative）。对一个二分问题来说，会出现四种情况。如果一个实例是正类并且也被预测成正类，即为真正类（True positive）,如果实例是负类被预测成正类，称之为假正类（False positive）。相应地，如果实例是负类被预测成负类，称之为真负类（True negative）,正类被预测成负类则为假负类（false negative）。


                  预测1                     预测0
      实际1        True Positive(TP)        False Negative(FN)
      实际0        False Positive(FP)        True Negative(TN)


TP：正确肯定的数目（本质是正例）；
FN：漏报，没有正确找到的匹配的数目（本质是正例）；
FP：误报，给出的匹配是不正确的（本质是负例）；
TN：正确拒绝的非匹配对数（本质是负例）；


## TPR、FPR、TNR

真正类率(True Positive Rate, TPR), 也称为Sensitivity，计算公式为
TPR = TP / (TP + FN).
计算的是分类器所正确识别出的正实例占所有正实例的比例。意味着正例里有多少被合理召回了。

负正类率(False Positive Rate, FPR),，也称为1-Specificity，计算公式为
FPR = FP / (FP + TN).
计算的是分类器错认为正类的负实例占所有负实例的比例。意味着负例里有多少被失误召回了。

真负类率（True Negative Rate，TNR），也称为specificity，计算公式为
TNR = TN /(FP + TN) = 1 - FPR.
计算的是分类器所正确识别出的负实例占所有负实例的比例。


## 准确率Accuracy、精确率Precision、召回率Recall

准确率是想要计算所有被分类器预测过的样本中，有多少比例是正确预测的。

准确率 Accuracy = 预测对的 / 所有

也可以计算为 Accuracy = (TP + TN) / (TP + TN + FP + FN).



我们认为分类器试图选出正例，那么精确率意味着其所选出的”正例“中有多少占比确实是对的？

精确率 Precision = 提取出的正确正例条数 / 提取出的”正例“信息条数    

精确率也可以计算为 Precision = TP / (TP + FP)




召回率又叫查全率，经常在搜索中使用，这一类场景我们更关心分类器对正例的判断结果，那么召回率意味着样本中的所有正例有多少占比被分类器给选出来了？

召回率 Recall = 提取出的正确信息条数 /  样本中的正确信息条数    

召回率也可以计算为 Recall = TP / (TP + FN)

这里可以看出，召回率Recall 和 真正例率Sensitivity（TPR） 是相同的。





# 错误分析 Error Analysis

偏差Bias，反映的是模型在样本上的输出与真实值之间的误差，即模型本身的精准度，即算法本身的拟合能力。偏差是模型本身导致的误差，即错误的模型假设所导致的误差，它是模型的预测值的数学期望和真实值之间的差距。


方差Variance，反映的是模型每一次输出结果与模型输出期望之间的误差，即模型的稳定性，反应预测的波动情况。方差是由于对训练样本集的小波动敏感而导致的误差。它可以理解为模型预测值的变化范围，即模型预测值的波动程度。


噪声Noise，是真实世界数据中的杂质，如果模型过度追求Low Bias会导致训练过度，对训练集判断表现优秀，导致噪声点也被拟合进去了。

总的来讲，模型的错误原因可以看做是 Error = Bias^2 + Variance + Noise

## 偏差分析


## 方差分析 

方差分析是机器学习中常用的来衡量模型对数据拟合好坏的度量方式。




# 假设检验

统计假设检验为模型的表现好坏提供了重要依据。

* 零假设和备选假设
* 检验类型： 单样本检验、独立双样本检验、相关配对检验
* 抽样分布类型：
* 检验方向： 单尾、双尾





# 分类模型评价指标 Classification Metric

## F1 Score

为了能够评价不同算法的优劣，在Precision和Recall的基础上提出了F1值的概念，来对Precision和Recall进行整体评价。

F1的定义如下：

F1值  = 正确率 * 召回率 * 2 / (正确率 + 召回率)

简单来讲就是，精确率和召回率的调和均值（倒数平均数的倒数）。

从评价来讲，精确率和准确率都高的情况下，F1 值也会高。


## AUC Area under Curve

AUC metric is only designed for binary classifiers.

机器学习实践中分类器常用的评价指标就是auc，不想搞懂，简单用的话，记住一句话就行。auc取值范围[0.5, 1]，越大表示越好，小于0.5的把结果取反就行。

roc曲线下的面积就是auc，所以要先搞清楚roc - receiver operating characteristic curve。
ROC曲线是以横坐标为 FPR ，以纵坐标为 TPR ，划出的一条曲线。
ROC曲线越靠拢(0,1)点，越偏离45度对角线越好，Sensitivity、Specificity越大效果越好。

AUC(Area under Curve)：Roc曲线下的面积，显然这个面积的数值不会大于1。


## P-R 曲线

P：查准率 ， R：查全率， 以R为横坐标，P为纵坐标，随着阈值改变，我们将得到P-R曲线。

## Classification Accuracy


## Logarithmic Loss

Logarithmic loss (related to cross-entropy) measures the performance of a classification model where the prediction input is a probability value between 0 and 1. The goal of our machine learning models is to minimize this value. A perfect model would have a log loss of 0. Log loss increases as the predicted probability diverges from the actual label.


## Confusion Matrix

可以画出一张二维矩阵， 横坐标为预测类别，纵坐标为真实类别，把所有样本的预测结果都填进矩阵里。
通过Confusion Matrix可以有个直观的感知。并且可以通过它计算出很多其他指标。
https://en.wikipedia.org/wiki/Confusion_matrix

![]({{site.baseurl}}/images/confusion_matrix.png)

```
from sklearn.metrics import confusion_matrix
import pandas as pd

confusion_df = pd.DataFrame(confusion_matrix(y_true,y_pred),
             columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
             index = ["Class " + str(class_name) for class_name in [0,1]])

print(confusion_df)
```

```
         Predicted Class 0  Predicted Class 1
Class 0                 14                  1
Class 1                  2                 13

```


## Classification Report

```
from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))
```

```
             precision    recall  f1-score   support

          0       0.88      0.93      0.90        15
          1       0.93      0.87      0.90        15

avg / total       0.90      0.90      0.90        30
```









# 回归模型评价指标 Regression Metric

Mean Absolute Error - MAE

Mean Squared Error - MSE

root-mean-square error (RMSE)

R^2 Metric











# 聚类模型评价指标 Clustering Metric

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


## MAP - Mean Average Precision

Mean average precision for a set of queries is the mean of the average precision scores for each query.

其中，Precision at i is a percentage of correct items among first i recommendations.



## NDCG - Normalized Discounted Cumulative Gain

DCG - Discounted cumulative gain

因为不同的搜索结果的数量很可能不相等，所以不同搜索的DCG值不能直接做对比。


## MRR - Mean Reciprocal Rank

是一个国际上通用的对搜索算法进行评价的机制，即第一个结果匹配，分数为1，第二个匹配分数为0.5，第n个匹配分数为1/n，如果没有匹配的句子分数为0。最终的分数为所有得分之和。











# 关联规则模型评价指标 Targeting model (association rule) Metric

关联规则希望推导一个 X -> Y 的规则，其中X、Y是itemset。集合T是所有已经发生的事件。

## 支持度(SUPPORT)

支持度表示项集{X,Y}在总项集里出现的频率。出现频率多的话，才能有足够多的样本来证明规则。

## 置信度(CONFIDENCE)

置信度表示在先决条件X发生的情况下，由关联规则”X→Y“推出Y的概率。指的是关联规则发生的频率。

## 提升度(LIFT)

提升度表示X->Y的关联是否是巧合。

如果Lift(X→Y)>1，则规则“X→Y”是有效的强关联规则。意思是X的发生会引起Y的发生。
如果Lift(X→Y) <=1，则规则“X→Y”是无效的强关联规则。意思是X的发生会引起Y的不发生。
特别地，如果Lift(X→Y) =1，则表示X与Y相互独立。


## 确信度(Conviction)

计算X发生且Y不发生的频率。


## 杠杆率(Leverage)








# 模型稳定性监控

对模型表现的监控指标：
0、Confusion Matrix
1、AUC(binary)
2、KS(binary)
3、PSI
4、Lift & Gain
5、MSE(Regression)

对业务信息的监控指标：
1、评分监控（评分模型）
2、响应率监控
3、模型变量监控（缺失值，平均值，最大值，最小值等，变量分布）
4、模型调用次数




# References

scikit-learn中的metric http://scikit-learn.org/stable/modules/model_evaluation.html

https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)

https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/

https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234

https://en.wikipedia.org/wiki/Association_rule_learning#Useful_Concepts

http://joshlawman.com/metrics-classification-report-breakdown-precision-recall-f1/
