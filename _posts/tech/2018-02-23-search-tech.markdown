---
title: "搜索技术入门"
categories: [Tech]
layout: post
---

搜索是计算机领域的一门学问。从早先的本地文件搜索，到互联网的网页搜索，再到垂直领域的内容搜索。
总之搜索的目的是在大量数据之中找东西。

从使用者的角度来看，搜索的输入是一段信息（字节串或关键字），搜索的输出是与输入信息对应的信息（一个或多个）。

从搜索引擎的角度来看，内部需要构建数据仓库的索引，在计算搜索结果时需要对多个结果进行排序。

常见的搜索往往不是唯一性的搜索那么直截了当，不像在SQL里select一个主键，找到就是有，找不到就是没有。
搜索输入不是简单的一个id或者hash，而是文字、图像或者语音。因此结果就不是唯一的是与非。
而是模棱两可的，需要搜索引擎实现者来分析哪些会是用户想找到的。


# 常见术语

Query：
指的是用户为了查询所输入的内容。一般是字符串，在图像搜索里就是图片，在语音搜索里就是语音数据。

Document：
指的是被查询对象中的一个单元。在网页搜索引擎中就是一个Web Page，在商品搜索里是一个商品sku，在图像、语音搜索里就是一张图片、一段音频。

正排索引：
以doc id为索引。每行是一个doc的内容（word的集合）。

倒排索引：
以query word为索引。每行是包含一个query word的doc链表。

TF-IDF：
Term Frequency - Inverse Document Frequency的缩写，即“词频-逆文本频率”。它由两部分组成，TF和IDF。

ad hoc：
前者类似于图书馆里的书籍检索，即书籍库(数据库)相对稳定不变，不同用户的查询要求是千变万化的。这种检索就称为ad hoc。基于Web的搜索引擎也属于这一类。

routing：
在routing中，查询常常称为profile，也就是通常所说的兴趣，用户的兴趣在一段时间内是稳定不变的，但是数据库(更确切的说，是数据流)是不断变化的。这种任务很象我们所说的新闻定制什么的，比如用户喜欢体育，这个兴趣在一段时间内是不变的，而体育新闻在不断变化。


# 查询项和搜索的内容

搜索的内容：
- 网页搜索
- 商品搜索
- 图片搜索

查询项：
- 查询词
- 一张图片
- 一条声音


# 搜索引擎的组成

简单来讲，搜索引擎中的一个查询过程是：

Step1（NLP）：在分词系统对用户请求等原始Query进行分析，产生对应的terms；

Step2(召回)：terms在倒排索引中的词项列表中查找对应的terms的结果列表；

Step3（IR）：对结果列表数据进行微运算，如：计算文档静态分，文档相关性等；

Step4（排序）：基于上述运算得分对文档进行综合排序，最后返回结果给用户；


## 分词、语义理解

来自用户的查询可能是一段话，或者多个次连在一起。首先需要通过分词来把原始的query划分为若干个独立的query词。

中心词。

同义词。

相关分类。

错字纠正。

搜索提示。

## 意图识别

比分词、语义理解更深一层的就是，要识别出用户搜索的意图。
比如搜索“苹果”，用户到底是在找苹果公司的信息，还是找吃的苹果的信息。这就需要结合用户的搜索上下文或者行为习惯来进行意图的识别。



## 数据构建

长期数据
实时数据


## 数据存储

首先所有原始文档信息的数量级非常的大，不仅需要分布式存储，连数据更新也会很麻烦，更别说搜索了。因此要考虑原始数据的存储和建立查询索引。由于数据量太大并且需要频繁更新，数据索引的量级也会非常大。所以索引一般也分成全量索引和增量索引。

## 数据索引

倒排索引结构。

为了提高查询速度，我们可以为特定的query词建立倒排索引。key为query词，值是包含该query词、或者与该query词具有相关性的所有doc的列表。


## 召回

业界比较流行是使用多种简单策略叠加合并的多路召回策略。

- 按分词召回
- 按向量相似度召回
- 个性化召回：协同过滤

## 文档相关性分析

主要的内容就是文本，也就是文本相关性。

## 文档重要性分析

只相关而不重要是不行的。这里的重要是可以包含很多含义的，总的来讲就是把更好的文档排在前面。

## 搜索过滤选项

搜索时用户选定了搜索范围、排名规则、筛选项。

## 搜索排序

一个词条的倒排索引中本身可能就是包含一定顺序的，不过这是一个大而全、以词频为基础的集合。对相关性，人气、质量、业务规则等因素考虑的不多。所以就有了排序模型专门对文档进行重排序。

排序本身是很大一块内容。对于搜索引擎来说，排序是对索引模块搜索到的文档进行排序。对于其他的一些常见，比如评论排序、答案排序，排序就是对所有文档进行排序。

## 业务干预

干预召回
干预排序




# Recall召回算法

WAND算法








# Rank排序算法

排序打分的三类方法：

    • Point-wise 每一个文章计算一个绝对得分，然后按照得分排序。
    • Pair-wise 每两个文章计算一下谁的得分高，用这个相对得分进行排序。
    • List-wise 枚举topk的所有排列情况，计算综合得分最高的一种作为结果。

从这三类方法都能看出，打分是排序的一个重要手段。

1.Pointwise方法主要包括以下算法：Pranking (NIPS 2002), OAP-BPM (EMCL 2003), Ranking with Large Margin Principles (NIPS 2002), Constraint Ordinal Regression (ICML 2005)。

2.Pairwise方法主要包括以下几种算法：Learning to Retrieve Information (SCC 1995), Learning to Order Things (NIPS 1998), Ranking SVM (ICANN 1999), RankBoost (JMLR 2003), LDM (SIGIR 2005), RankNet (ICML 2005), Frank (SIGIR 2007), MHR(SIGIR 2007), Round Robin Ranking (ECML 2003), GBRank (SIGIR 2007), QBRank (NIPS 2007), MPRank (ICML 2007), IRSVM (SIGIR 2006)  

3.Listwise方法主要包括以下几种算法：LambdaRank (NIPS 2006), AdaRank (SIGIR 2007), SVM-MAP (SIGIR 2007), SoftRank (LR4IR 2007), GPRank (LR4IR 2007), CCA (SIGIR 2007), RankCosine (IP&M 2007), ListNet (ICML 2007), ListMLE (ICML 2008) 。


## 基于投票的排序算法

-- 威尔逊得分排序

一种基于用户投票的排名算法算法。用于质量排序，数据含有好评和差评，综合考虑评论数与好评率，得分越高，质量越高。

![Wilson score interval](https://wikimedia.org/api/rest_v1/media/math/render/svg/c999019cc4f806d147bffb46158dc022b822f01a)

其中，u表示正例数（好评），v表示负例数（差评），n表示实例总数（评论总数），p表示好评率，z是正态分布的分位数（参数），S表示最终的威尔逊得分。z一般取值2即可，即95%的置信度。


-- Reddit网站的热帖排序

Reddit网站对热门帖子排序考虑了帖子的存在时间t，赞成票与反对票的差值x。

![reddit](http://chart.googleapis.com/chart?cht=tx&chl=Score%3Dlog_%7B10%7Dz%2B%5Cfrac%7Byt%7D%7B45000%7D&chs=80)

其中，z表示赞成票与反对票之间差额的绝对值。如果对某个帖子的评价，越是一边倒，z就越大。如果赞成票等于反对票，z就等于1。y表示对文章的总体看法。如果赞成票居多，y就是+1；如果反对票居多，y就是-1；如果赞成票和反对票相等，y就是0。


-- HackerNews网站的文章排序

HackerNews热门文章的排序考虑了文章的赞成票数，帖子的时间衰减。

![HN](http://chart.googleapis.com/chart?cht=tx&chl=Score%3D%5Cfrac%7BP-1%7D%7B(T%2B2)%5E%7BG%7D%7D&chs=60)

其中，P表示帖子的赞成票数，减去1是为了忽略发帖人的投票。T表示帖子存在的时间，加上2是为了防止最新的帖子导致分母过小（之所以选择2，可能是因为从原始文章出现在其他网站，到转贴至Hacker News，平均需要两个小时）。G表示"重力因子"（gravityth power），即将帖子排名往下拉的力量，默认值为1.8。


-- StackOverflow的热门问答排序

StackOverflow的热门问答排序考虑了用户的浏览次数、对问题和回答的投票、以及时间衰减。

![StackOverflow](http://chart.googleapis.com/chart?cht=tx&chl=%5Cfrac%7B(log_%7B10%7DQviews)%5Ctimes%204%2B%5Cfrac%7BQanswers%5Ctimes%20Qscore%7D%7B5%7D%2Bsum(Ascores)%7D%7B((Qage%2B1)-(%5Cfrac%7BQage-Qupdated%7D%7B2%7D))%5E%7B1.5%7D%7D&chs=120)

其中，Qviews代表问题的浏览次数。以10为底的对数，用意是当访问量越来越大，它对得分的影响将不断变小。Qanswers代表问题中的答案数量。Qscore代表问题得分=对问题的赞成票-反对票。Ascores代表一个回答的得分=对回答的赞成票-反对票。Qage代表问题的存在时间。Qupdated代表最后一个回答的存在时间。


-- IMDB的电影评分排序

IMDB电影打分考虑了用户对一部电影的投票和所有电影的平均票数。把m个平均分票加给冷门电影以保证它有足够多的票数。

![imdb](http://chart.googleapis.com/chart?cht=tx&chl=WR%3D%5Cfrac%7Bv%7D%7Bv%2Bm%7DR%2B%5Cfrac%7Bm%7D%7Bv%2Bm%7DC&chs=60)

其中，R表示投票分数的平均数。V表示投票人数。m表示排名前250名的电影的最低投票数。C表示所有电影的平均得分。


## 基于目标的排序算法

前面的基于投票的排序都是根据投票判断一个文档的好坏程度来打分。对于没有投票的场景，比如订阅流、推荐流，希望是结合更多维度的特征来判断，目标也不仅仅是好坏程度，而可能是用户的感兴趣程度。

对于这种排序，假设要考虑几十种因素来打分。这里直接把考虑因素叫作特征吧。有些情况下，特征本身就是需要计算间接得到的。比如EdgeRank算法。

-- Facebook的新鲜事排序EdgeRank

EdgeRank用来给一个用户的新鲜事流进行排序。计算公式为

![EdgeRank](https://beta.techcrunch.com/wp-content/uploads/2010/04/edgerankform2.png)

其中，u表示事件生产者和观察者之间的亲密度。w表示由事件类型所决定的权重（创建、点赞、评分、转发等不同类型事件的权重不同）。d表示时间衰减因子。


有些情况下，直接设计一个打分的公式算法不太现实。这时就可以借助机器学习方法来拟合一个从考虑因素到目标分数的函数。一般叫做Learn to Rank，通过训练数据来学出一个排序算法（排序模型）来。


### RankNet

RankNet算法是一种复杂度为 O(n) 但是属于pair-wise方法的算法。它是基于模型的算法，可以是神经网络模型也可以是其他模型。

训练目标：以错误pair最少为优化目标，错误pair也被称作是inversion。
优化方法：使用Stochastic Gradient Descent作为优化方法。

http://x-algo.cn/index.php/2016/07/31/ranknet-algorithm-principle-and-realization/

https://blog.csdn.net/huagong_adu/article/details/40710305


### LambdaRank

基于RankNet之上，我们发现其实不需要计算损失，只需损失的梯度即可对排序目标进行评价。
LambdaRank使用了新的损失函数：

LambdaRank是一个经验算法，它不是通过显示定义损失函数再求梯度的方式对排序问题进行求解，而是分析排序问题需要的梯度的物理意义，直接定义梯度，即Lambda梯度。

https://www.cnblogs.com/bentuwuying/p/6690836.html

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.180.634&rep=rep1&type=pdf


### LambdaMART

结合了LambdaRank 和 MART （Multiple Additive Regression Trees）。

这篇论文是对以上三种排序模型的详细介绍。
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf




# Rerank重排序算法

## 什么是Rerank？ 为什么已经有Rank了还要再Rerank呢？
重排主要是对排序后结果的优化，也可以用于二次推荐。

Rank阶段，模型以单个item为预测目标，排序时不会兼顾到多个item之间的相互关系或影响。

rerank(重排) 跟Rank的区别在于，rank对每个商品独立打分，而rerank 结合所有商品给出整体分数, 相当于listwise的排序思想。

## ReRank一般做什么事？

- 去重
- item多样性
- 业务的特殊需求
- Exploitation Exploration
- 强化学习

Seq2Slate











# 开源搜索引擎框架

如果不是有极为特殊的业务需求，或者数据量极为庞大，通用的开源搜索引擎就可以满足一般场景的搜索需求。

## Lucene


## Elasticsearch

Rest API

Status

        http://localhost:9200/_cat/health?v
        http://localhost:9200/_cat/nodes?v
        http://localhost:9200/_cat/indices?v

Index

一个index是具有某种相似属性的文档的集合，index以一个名字来标识。


添加一个index

        PUT http://localhost:9200/customer?pretty

        PUT http://localhost:9200/customer/_doc/1?pretty  header里要设置content-type为json

        http://localhost:9200/customer/_doc/1?pretty

删除一个index

        DELETE http://localhost:9200/customer?pretty


Search

        http://localhost:9200/customer/_search


## solr



# 开源排序框架


## RankLib

RankLib 不是一个搜索框架，而是一个排序算法库，提供了常见的几种排序算法。

MART, RANKBOOST, RANKNET, ADARANK, COOR_ASCENT, LAMBDARANK, LAMBDAMART, LISTNET, RANDOM_FOREST

https://sourceforge.net/p/lemur/wiki/RankLib/


## TensorFlow Ranking

有一款排序算法库。

https://github.com/tensorflow/ranking






# References

电商搜索引擎的架构设计和性能优化 http://www.csdn.net/article/2015-10-29/2826086

京东亿级商品搜索核心技术解密 http://geek.csdn.net/news/detail/126786

https://tech.youzan.com/you_zan_searchengine2/

http://www.36dsj.com/archives/61886

电商搜索引擎报告 https://www.jianshu.com/p/aaed2650ee18

美团的搜索排序方案 https://tech.meituan.com/meituan-search-rank.html

信息检索评价指标 https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision
