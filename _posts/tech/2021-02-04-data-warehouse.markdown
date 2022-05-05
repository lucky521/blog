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

* DWD - Data Warehouse Detai 数据明细层，

* DWM - Data WareHouse Middle 数据中间层，轻度汇总层，明细层需要复杂清洗的数据和需要MR处理的数据也经过处理后接入到轻度汇总层。

* DWS - Data WareHouse Servce

* APP - 数据应用层



# Join背后有哪些种实现方式？

* Nested-loop (NL) join 暴力双循环遍历
* Block nested-loop (BNL) join 先按key分块，块内再暴力遍历
* Hash join 小表和大表join，把小表做成hashmap放在内存，遍历大表。 功能退化为只支持相等条件的join。
* Grace hash join  先按key分块，块内再进行hash join
* Broadcast hash join 在分布式场景，把小表广播到每一个executor节点，执行hash join
* Shuffle hash join 在分布式场景，把小表shuffle分块发送到不同的executor节点，执行hash join
* Sort-merge join  先对两个表排序，然后比较。如果join的key本来就有序，这就很快了


# 参考

[终于有人将数据湖讲明白了](https://view.inews.qq.com/a/20210909A0ARO400)

数据湖的价值
* 流批作业在数据层面做到进一步的统一
  * 快速upsert
  * table schema
* Time Travel Query

数据库、数据仓库、数据湖