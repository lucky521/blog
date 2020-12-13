---
title: "大数据处理的方法"
subtitle: "Working on Big Data"
categories: [Language]
layout: post
---

本篇想要总结一下在Hadoop集群上处理大数据文件的方法，主要从应用层面去看我们有多少种、怎么样的方式（写什么样的程序、越简单越好的程序）去读写大数据进行增删查改，不打算关注大数据概念，也不关注框架的实现部分。



# Hive

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


# Pig


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

# Hadoop Streaming

https://hadoop.apache.org/docs/r1.2.1/streaming.html


# Python MapReduce

https://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/



# Spark 

## PySpark

## Spark SQL

```java
    Dataset<Row> df = sparkSession.sql(cmdLine.getSql());
```



# Presto

presto是Facebook开源的大数据查询引擎，为了解决hive查询慢产生
