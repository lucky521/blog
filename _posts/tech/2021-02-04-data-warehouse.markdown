---
title: "数仓技术"
subtitle: "Data Warehouse"
categories: [Tech]
layout: post
---


## 数据模型
数据模型所描述的内容包括三个部分：数据结构、数据操作、数据约束

* 星型模型
* 雪花模型

## 分层概念

* 数据来源层， 一般为业务数据库，埋点数据。

* ODS - Operational Data Store 数据运营层， 数据从源表拉过来，进行ETL，产出的数据写入，就是写入到ODS层。

* DWD - Data Warehouse Detai 数据明细层，

* DWM - Data WareHouse Middle 数据中间层，轻度汇总层，明细层需要复杂清洗的数据和需要MR处理的数据也经过处理后接入到轻度汇总层。

* DWS - Data WareHouse Servce

* APP - 数据应用层




# 参考

[终于有人将数据湖讲明白了](https://view.inews.qq.com/a/20210909A0ARO400)

数据湖的价值
* 流批作业在数据层面做到进一步的统一
  * 快速upsert
  * table schema
* Time Travel Query

数据库、数据仓库、数据湖