---
title: "大数据处理框架"
subtitle: "Big Data Architecture"
categories: [Tech]
layout: post
---

大数据和分布式这两个概念总是会混在一起。本文主要以大数据为主题，会有意识的不谈分布式系统。

大数据的大有两个角度，一种是静态数据数据量巨大；一种是流式数据数据永无止境。


# Schedule
## MR
https://www.netjstech.com/p/hadoop-framework-tutorial.html

map运行阶段分为:Read、Map、Collect、Spill、Merge五个阶段。
reduce 运行阶段分为shuflle(copy) merge  sort    reduce write五个阶段。

## Yarn

yarn application -list -appStates ALL | grep "rank"  | wc -l

yarn application -status  application_9173934103802_23474815

hadoop job -list   | grep "rank" |  awk '{split($0,a,"\t"); print a[1],a[5],a[7],a[9],a[12]}'


# Storing

Data lake vs data warehouse， 数据湖和数据仓库的比较
数据湖的一个特点是，存储的数据没有预先设定schema，保存原始数据。
注意数据湖不是个项目，而是一个概念和思想。

Redis

HBase

Pika  https://github.com/Qihoo360/pika

https://github.com/delta-io/delta


Hudi

* 快速upsert，可插入索引
* 可原子操作
* 有savepoint
* 管理文件大小

用spark-shell访问hudi





# Streaming

## streams && tables


flume - 把来自不同源头不同节点的大量数据发送到中心存储。

kafka 
http://kafka.apache.org/quickstart

原生版本： https://github.com/edenhill/librdkafka
C++版本：  https://github.com/mfontanini/cppkafka
Python:   https://github.com/confluentinc/confluent-kafka-python

MQ




# Computing

Storm

## Spark

spark是怎么工作的？
RDD
Lazy Evaluation


## Flink




# OLAP

OLAP场景的关键特征
* 绝大多数是读请求
* 数据以相当大的批次(> 1000行)更新，而不是单行更新;或者根本没有更新。
* 已添加到数据库的数据不能修改。
* 对于读取，从数据库中提取相当多的行，但只提取列的一小部分。
* 宽表，即每个表包含着大量的列
* 查询相对较少(通常每台服务器每秒查询数百次或更少)
* 对于简单查询，允许延迟大约50毫秒
* 列中的数据相对较小：数字和短字符串(例如，每个URL 60个字节)
* 处理单个查询时需要高吞吐量(每台服务器每秒可达数十亿行)
* 事务不是必须的
* 对数据一致性要求低
* 每个查询有一个大表。除了他以外，其他的都很小。
* 查询结果明显小于源数据。换句话说，数据经过过滤或聚合，因此结果适合于单个服务器的RAM中

ES https://www.elastic.co/start

ClickHouse https://clickhouse.tech/

Doris https://doris.apache.org/master/en/

druid https://druid.apache.org/

kylin http://kylin.apache.org/

https://www.hologres.io/

https://kudu.apache.org/

https://iceberg.apache.org/

TiDB https://github.com/pingcap/tidb



# 参考

https://github.com/wangzhiwubigdata/God-Of-BigData