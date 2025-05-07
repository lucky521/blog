---
title: "数仓技术"
subtitle: "Data Warehouse"
categories: [Tech]
layout: post
---


# 数据模型
数据模型所描述的内容包括三个部分：数据结构、数据操作、数据约束

* 星型模型
* 雪花模型

# 分层概念

* 数据来源层， 一般为业务数据库，埋点数据。
* ODS - Operational Data Store 数据运营层， 数据从源表拉过来，进行ETL，产出的数据写入，就是写入到ODS层。
* DWD - Data Warehouse Detail 数据明细层，
* DWM - Data WareHouse Middle 数据中间层，轻度汇总层，明细层需要复杂清洗的数据和需要MR处理的数据也经过处理后接入到轻度汇总层。
* DWS - Data WareHouse Servce
* ADS or APP - 数据应用层



# Join背后有哪些种实现方式？

* Nested-loop (NL) join 
暴力双循环遍历
* Block nested-loop (BNL) join 
先按key分块，块内再暴力遍历
* Hash join 小表和大表join
把小表做成hashmap放在内存，遍历大表。 功能退化为只支持相等条件的join。
* Grace hash join  
先按key分块，块内再进行hash join
* Broadcast hash join 
在分布式场景，把小表广播到每一个executor节点，执行hash join
* Shuffle hash join 
在分布式场景，把小表shuffle分块发送到不同的executor节点，执行hash join
* Sort Merge join  
先对两个表排序，然后比较。如果join的key本来就有序，这就很快了


## 大表join优化思路
尺寸较大的称为外表，尺寸较小的称为内表
* 分而治之：
  1. 内表按行拆分
  2. 我们让外表依次与内表子集做关联，得到部分计算结果。
  3. 最后，再用Union操作把所有的部分结果合并到一起

# 湖仓一体
***数据库、数据仓库、数据湖***

简单来说，数据湖是一个存储大量原始数据的地方，可以存放各种格式的数据，而数据仓库则是经过处理和整理的数据存储，适合进行分析和查询。湖仓一体的核心思想是把这两者的优点结合起来。

[终于有人将数据湖讲明白了](https://view.inews.qq.com/a/20210909A0ARO400)

数据湖的价值
* 流批作业在数据层面做到进一步的统一
  * 快速upsert
  * table schema
* Time Travel Query 时间旅行 、带时间版本查询、Stale Read
