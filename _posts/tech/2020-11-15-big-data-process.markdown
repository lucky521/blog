---
title: "大数据处理框架"
subtitle: "Big Data Architecture"
categories: [Tech]
layout: post
---

大数据和分布式这两个概念总是会混在一起。本文主要以大数据为主题，会有意识的不谈分布式系统。

大数据的大有两个角度，一种是静态数据数据量巨大；一种是流式数据数据永无止境。


# Resource

## Yarn

yarn application -list -appStates ALL | grep "rank"  | wc -l

yarn application -status  application_9173934103802_23474815

hadoop job -list   | grep "rank" |  awk '{split($0,a,"\t"); print a[1],a[5],a[7],a[9],a[12]}'

## K8s





# Storing

## Hbase
HDFS是Hadoop的存储系统，它的优点是可以存储超大量数据，但是缺点是速度慢。
HBase建立在HDFS之上，以KV的形式存储，提供实时访问。
HBase 本身只提供了Java 的API 接口。
snapshot是HBase非常核心的一个功能，使用snapshot的不同用法可以实现很多功能

## Storage Cache

alluxio

EVCache https://github.com/Netflix/EVCache


## Data Format

数据格式： parquet, avro, orc, csv, json

数据压缩： zstd, brotli, lz4, gzip, snappy, uncompressed


## Data Lake数据存储中间Table format层

Data lake vs data warehouse， 数据湖和数据仓库的比较
数据湖的一个特点是，存储的数据没有预先设定schema，保存原始数据。
注意数据湖不是个项目，而是一个概念和思想。

具备 ACID 能力的表格式中间件:

* hudi https://github.com/apache/hudi
* iceberg https://iceberg.apache.org/
* deltalake https://github.com/delta-io/delta

## Hudi

Hudi 是 Uber 主导开发的开源数据湖框架.
增量模型，简单来讲，是以 mini batch 的形式来跑准实时任务。Hudi 在增量模型中支持了两个最重要的特性，

Hudi表的数据文件，可以使用操作系统的文件系统存储，也可以使用HDFS这种分布式的文件系统存储。为了后续分析性能和数据的可靠性，一般使用HDFS进行存储。

* 快速upsert，可插入索引
* 可原子操作
* 有savepoint
* 管理文件大小

* Copy-On-Write Table : 在写文件的时候就做了数据合并,因此写入数据的压力比较大, 对读数据比较友好.
* Merge-On-Read Table : 在读数据的时候合并, 写入是数据采用append的方式,适合快速写入的场景.

* preCombineField 属性用于在主键重复时合并数据。 若设置了该字段，upsert操作，有预合并， 当主键重复时，去重保留preCombineField字段最大的记录



## Iceberg

在不影响已存在数据使用体验的情况下支持以下特性：
* Table Schema支持add、drop、rename、update type、reorder
* Table Partition支持变更
* Table Sort Order支持变更



# Streaming 数据流动
## streams && tables

flume - 把来自不同源头不同节点的大量数据发送到中心存储。

kafka - http://kafka.apache.org/quickstart

原生版本： https://github.com/edenhill/librdkafka
C++版本：  https://github.com/mfontanini/cppkafka
Python:   https://github.com/confluentinc/confluent-kafka-python

生产者和消费者以极高的速度生产/消费大量数据或产生请求，从而占用broker上的全部资源，造成网络IO饱和。有了配额（Quotas）就可以避免这些问题。Kafka支持配额管理，从而可以对Producer和Consumer的produce&fetch操作进行流量限制，防止个别业务压爆服务器。

MQ



# Schedule Computing 批处理任务
纯离线调度数据任务。
## MR
https://www.netjstech.com/p/hadoop-framework-tutorial.html

map运行阶段分为:Read、Map、Collect、Spill、Merge五个阶段。
reduce 运行阶段分为shuflle(copy) merge  sort reduce write五个阶段。

每个阶段的数据都要落磁盘，因而数据量无论多大都能搞，因此也特别慢。

## Tez
本质上还是基于mr，算是对mr做了dag方向的优化


## Spark




# Stream Computing 流处理任务

## Storm

## Spark Streaming

spark是怎么工作的？
RDD - Resilient Distributed Dataset
lineage
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
* SETTINGS storage_policy = 'jdob_ha';
* ENGINE = Distributed(xxx, xxx, xxx, rand());

Doris https://doris.apache.org/master/en/

druid https://druid.apache.org/

kylin http://kylin.apache.org/

hologres https://www.hologres.io/

kudu https://kudu.apache.org/


##  HTAP

TiDB https://github.com/pingcap/tidb





# 参考

https://github.com/wangzhiwubigdata/God-Of-BigData