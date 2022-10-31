---
title: "大数据处理的语言和方法"
subtitle: "Working on Big Data"
categories: [Language]
layout: post
---

本篇想要总结一下在Hadoop集群上处理大数据文件(批式静态数据)的方法，主要从应用层面去看我们有多少种、怎么样的方式（写什么样的程序、越简单越好的程序）去读写大数据进行增删查改，不打算关注大数据概念，也不关注框架的实现部分。

不同引擎虽虽然可能都是SQL，但底层执行方法是有一定差异的，比如spark/presto直接读取文件解析，hive根据元数据mapping解压；比如spark/hive对空值、空字符串的判断处理和presto不一样。


# 原生 MapReduce 接口
https://hadoop.apache.org/docs/r2.10.2/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html

Mapper - setMapperClass

Reducer - setReducerClass
* 有三个主要阶段
  * shuffle
  * sort
  * reduce


Partitioner - setPartitionerClass


Combiner - setCombinerClass
* Combiner的作用就是对map端的输出先做一次合并，以减少在map和reduce节点之间的数据传输量。 它相当于是一个“迷你reduce”过程，它只处理单台机器生成的数据。
* 你要确定你的处理逻辑是否允许这样的局部提前聚合。默认情况下是不做的。
* 有些场景下combiner实现类可以跟reducer实现类相同


# Hadoop Streaming
https://hadoop.apache.org/docs/r1.2.1/streaming.html
用户编写读写终端stream的单机程序，由 Hadoop Streaming 框架将其组合成MR分布式程序。
stdin、stdout中的每一行表示一个key-value数据，默认由tab符间隔。


Python MapReduce
https://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/



# HiveSQL
hive是Java实现的，由Facebook开源，目的是将特定的SQL语句编译为MapReduce jar包扔给hadoop去执行，本质上是一个代码转换编译的工具，简化mr的开发。

https://ytluck.github.io/data-mining/my-dataming-post-42.html



## hive tblproperties 表属性
向表中添加自定义或预定义的元数据属性，并设置它们的赋值。在hive建表时，可设置TBLPROPERTIES参数修改表的元数据，也能通过ALTER语句对其修改。

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

## hive python udf

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


## Hive优化选项
hive-site.xml  :hive客户端配置和服务端配置都是这个文件

配置项： https://blog.csdn.net/u013760453/article/details/84580012

官网配置项大全： https://cwiki.apache.org/confluence/display/Hive/Configuration+Properties

* 选取执行引擎： mr tez spark  , set hive execution.engine=spark
  * Options are: mr (Map Reduce, default), tez (Tez execution, for Hadoop 2 only), or spark (Spark execution, for Hive 1.1.0 onward).
* 选取文件存储格式:
* 选取优化器: set  hive.cbo.enable=true



## Map join
sql中涉及到多张表的join，当有一张表的大小小于1G时，使用Map Join可以明显的提高SQL的效率。如果最小的表大于1G，使用Map Join会出现OOM的错误。

## skew join




# Pig

Pig也是Java实现的，由雅虎开源，使用类似于python的语法。思想和Hive一致，都是运行前将pig语言转换为MR job来运行。

## pig udf
```python
REGISTER 'my_udf.py' using jython as my_udf;
data = LOAD '$eval_table' using org.apache.hcatalog.pig.HCatLoader();

data = foreach data generate
   key_sku as key_sku,
   STRSPLIT(similar_sku_list, ',') as similar_sku_list,
   STRSPLIT(nn_sku_list, ',') as nn_sku_list;

result = FOREACH data generate my_udf.my_function(key_sku, similar_sku_list, nn_sku_list);

STORE result into '$result_filename' USING PigStorage('\t', '-schema');
```

```python
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





# PrestoSQL

presto是Java实现的， 由Facebook开源，为了解决hive查询慢产生。提供的用户语言也是SQL。

## trino 
2020年底，PrestoSQL更名为Trino


# Impala
impala由C++实现，提供的用户语言也是SQL。调用C语言层的libhdfs来直接访问HDFS处理数据。

impala 使用hive的元数据, 完全在内存中计算。 使用上和Presto很接近。


# Spark 
当前最流行的批处理计算引擎应该就是spark了。

* spark-shell 命令行交互开发
* spark-submit 提交jar包运行
* spark-sql 直接用sql交互开发

* Dataset API
* SQL API

## Spark scala/java

* SparkConf
* sparkContext , SparkSession 
* getOrCreate
* registerKryoClasses
* createOrReplaceTempView
* Dataset 
* DataFrame 执行SparkSession.sql("...")返回数据的类型， 等价于table的概念

```scala
    import org.apache.spark.sql.SparkSession

    val conf = new SparkConf().setAppName(getClassName)
    conf.set("spark.driver.cores","4")
    conf.set("spark.streaming.clean.shuffle.enabled","true")
    conf.set("spark.serializer","org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.executor.extraJavaOption","-XX:+UseG1GC")

    val spark = SparkSession
      .builder
      .enableHiveSupport()
      .config(conf)
      .getOrCreate

    spark.sql("")
         .repartition(100)
         .createOrReplaceTempView("xxx")

    spark.udf.register("xxx", )
```

* coalesce
* repartition
  * https://stackoverflow.com/questions/31610971/spark-repartition-vs-coalesce



## PySpark

## SparkSQL
SparkSQL引擎
Spark SQL can use existing Hive metastores, SerDes, and UDFs.

> spark-sql

```java
    Dataset<Row> df = sparkSession.sql(cmdLine.getSql());
```

## Hive on Spark
HiveSQL引擎


## Structured Streaming
***Structured Streaming provides fast, scalable, fault-tolerant, end-to-end exactly-once stream processing without the user having to reason about streaming.***

spark希望你以表达批式计算的方式一样去表达出流式计算。SQL引擎负责增量的、持续的去更新结果。其内部使用的是micro-batch处理方式。

```java
Dataset<Row> lines = spark
  .readStream()
  .format("socket")
  .option("host", "localhost")
  .option("port", 9999)
  .load();

// Split the lines into words
Dataset<String> words = lines
  .as(Encoders.STRING())
  .flatMap((FlatMapFunction<String, String>) x -> Arrays.asList(x.split(" ")).iterator(), Encoders.STRING());

// Generate running word count
Dataset<Row> wordCounts = words.groupBy("value").count();

StreamingQuery query = wordCounts.writeStream()
  .outputMode("complete")
  .format("console")
  .start();

query.awaitTermination();
```


## Spark Streaming




# Phoenix
Phoenix 基于Hbase给面向业务的开发人员提供了以标准SQL的方式对Hbase进行查询操作，并支持标准SQL中大部分特性:条件运算,分组，分页，等高级查询语法。




# Ref

[10 ways to query Hadoop with SQL](https://www.infoworld.com/article/2683729/10-ways-to-query-hadoop-with-sql.html)