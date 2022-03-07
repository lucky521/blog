---
title: "大数据处理的语言和方法"
subtitle: "Working on Big Data"
categories: [Language]
layout: post
---

本篇想要总结一下在Hadoop集群上处理大数据文件(批式静态数据)的方法，主要从应用层面去看我们有多少种、怎么样的方式（写什么样的程序、越简单越好的程序）去读写大数据进行增删查改，不打算关注大数据概念，也不关注框架的实现部分。


# Hive
hive是Java实现的，由Facebook开源，目的是将特定的SQL语句编译为MapReduce jar包扔给hadoop去执行，本质上是一个代码转换编译的工具，简化mr的开发。

## hive metastore

hive依赖 Metastore 服务作为hadoop和用户接口的纽带。 Metastore本质上是一个提供Thrift服务的元信息服务器。

https://cwiki.apache.org/confluence/display/Hive/AdminManual+Metastore+Administration

## 其他hive services

* Hive Services
* HiveServer2
* Hive Metastore
* HCatalog + WebHcat
* Beeline & Hive CLI
* Thrift client
* FileSystem :: HDFS and other compatible filesystems like S3
* Execution engine :: MapReduce, Tez, Spark
* Hive Web UI (added in Hive 2.x). Maybe also Tez or Spark UI, but not really*

## python udf

https://florianwilhelm.info/2016/10/python_udf_in_hive/

## 开窗函数


## hive SerDe
1、SerDe is a short name for “Serializer and Deserializer.”
Hive uses SerDe (and !FileFormat) to read and write table rows.
HDFS files –> InputFileFormat –> <key, value> –> Deserializer –> Row object
Row object –> Serializer –> <key, value> –> OutputFileFormat –> HDFS files
2、InputFormat是将数据读取进来，解析成一个个记录，SerDe通过Deserializer将记录解析成字段
3、SerDe通过Serializer将字段解析成一个个记录，再通过OutputFileFormat将记录写到存储系统

当面临一个HDFS上的文件时，Hive将如下处理（以读为例）：
(1) 调用InputFormat，将文件切成不同的文档。每篇文档即一行(Row)。
(2) 调用SerDe的Deserializer，将一行(Row)，切分为各个字段。

当HIVE执行INSERT操作，将Row写入文件时，主要调用OutputFormat、SerDe的Seriliazer，顺序与读取相反。

列式存储和压缩

Hive SQL背后的原理： https://tech.meituan.com/2014/02/12/hive-sql-to-mapreduce.html


## hive-site.xml hive客户端配置和服务端配置都是这个文件








# Pig

Pig也是Java实现的，由雅虎开源，使用类似于python的语法。思想和Hive一致，都是运行前将pig语言转换为MR job来运行。

## pig udf
```
REGISTER 'my_udf.py' using jython as my_udf;
data = LOAD '$eval_seq2seq_table' using org.apache.hcatalog.pig.HCatLoader();

data = foreach data generate
   key_sku as key_sku,
   STRSPLIT(similar_sku_list, ',') as similar_sku_list,
   STRSPLIT(nn_sku_list, ',') as nn_sku_list;

result = FOREACH data generate my_udf.my_function(key_sku, similar_sku_list, nn_sku_list);

STORE result into '$result_filename' USING PigStorage('\t', '-schema');
```

```
@outputSchema("t:(key_sku:chararray,bad:int)")
def my_function(key_sku, similar_sku_list, nn_sku_list):
    if len(nn_sku_list) <= 1:
        return (key_sku, 0)
    bad = 1
    for item in similar_sku_list:
        if key_sku == item:
            continue
        if item in nn_sku_list:
            bad = 0
            break
    return (key_sku, bad) 
```





# Presto

presto是Java实现的， 由Facebook开源，为了解决hive查询慢产生。提供的用户语言也是SQL。

# Impala

impala由C++实现，提供的用户语言也是SQL。调用C语言层的libhdfs来直接访问HDFS处理数据。

impala 使用hive的元数据, 完全在内存中计算。 使用上和Presto很接近。



# Hadoop Streaming

https://hadoop.apache.org/docs/r1.2.1/streaming.html


## Python MapReduce

https://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/



# Spark 


## PySpark

## Spark SQL

spark-sql

```java
    Dataset<Row> df = sparkSession.sql(cmdLine.getSql());
```


# Ref

[10 ways to query Hadoop with SQL](https://www.infoworld.com/article/2683729/10-ways-to-query-hadoop-with-sql.html)