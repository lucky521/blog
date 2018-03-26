---
title: "搜索技术入门"
categories: [design]
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



# 搜索引擎

## 分词与语义理解

来自用户的查询可能是一段话，或者多个次连在一起。首先需要通过分词来把原始的query划分为若干个独立的query词。

同义词。

相关分类。


## 数据存储


## 数据索引

倒排索引结构。

为了提高查询速度，我们可以为特定的query词建立倒排索引。key为query词，值是包含该query词、或者与该query词具有相关性的所有doc的列表。

## 内容相关性分析

主要的内容就是文本，也就是文本相关性。


## 搜索过滤选项

搜索选择决定了搜索范围、排名规则。


## 搜索排序

排序本身是很大一块内容。对于搜索引擎来说，排序是对索引模块搜索到的文档进行排序。对于其他的一些常见，比如评论排序、答案排序，排序就是对所有文档进行排序。

Learn to Rank

排序打分。






# 开源搜索引擎

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




# References

电商搜索引擎的架构设计和性能优化 http://www.csdn.net/article/2015-10-29/2826086

京东亿级商品搜索核心技术解密 http://geek.csdn.net/news/detail/126786

https://tech.youzan.com/you_zan_searchengine2/

http://www.36dsj.com/archives/61886

电商搜索引擎报告 https://www.jianshu.com/p/aaed2650ee18

美团的搜索排序方案 https://tech.meituan.com/meituan-search-rank.html
