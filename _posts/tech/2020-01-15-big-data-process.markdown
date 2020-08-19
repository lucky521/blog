---
title: "大数据处理框架"
subtitle: "Big Data Architecture"
categories: [Tech]
layout: post
---

大数据和分布式这两个概念总是会混在一起。本文主要以大数据为主题，会有意识的不谈分布式系统。

大数据的大有两个角度，一种是静态数据数据量巨大；一种是流失数据数据永无止境。


## Storing

Redis

HBase

Pika  https://github.com/Qihoo360/pika

## Streaming

flume - 把来自不同源头不同节点的大量数据发送到中心存储。

kafka 
http://kafka.apache.org/quickstart

原生版本： https://github.com/edenhill/librdkafka
C++版本： https://github.com/mfontanini/cppkafka

MQ



## Computing

Storm

Spark

Flink