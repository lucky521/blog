---
title: "大数据处理的语言和方法"
subtitle: "Working on Big Data"
categories: [Language]
layout: post
---

本篇想要总结一下在Hadoop集群上处理大数据文件(批式静态数据)的方法，主要从应用层面去看我们有多少种、怎么样的方式（写什么样的程序、越简单越好的程序）去读写大数据进行增删查改，不打算关注大数据概念，也不关注框架的实现部分。

不同引擎虽虽然可能都是SQL，但底层执行方法是有一定差异的，比如spark/presto直接读取文件解析，hive根据元数据mapping解压；比如spark/hive对空值、空字符串的判断处理和presto不一样。

SQL语言共分为四大类：
* 数据查询语言DQL
* 数据操纵语言DML
* 数据定义语言DDL
* 数据控制语言DCL


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

* Hive对于null类型也是比较敏感的，如果在select过程遇到空指针异常，注意你的数据中是否有超出预期的null。
* insert table的顺序非常重要(名字不重要)，务必跟建表时的字段顺序保持一致


## hive tblproperties 表属性
向表中添加自定义或预定义的元数据属性，并设置它们的赋值。在hive建表时，可设置TBLPROPERTIES参数修改表的元数据，也能通过ALTER语句对其修改。

## hive metastore

hive依赖 Metastore 服务作为hadoop和用户接口的纽带。 Metastore本质上是一个提供Thrift服务的元信息服务器。

https://cwiki.apache.org/confluence/display/Hive/AdminManual+Metastore+Administration


Hive Metastore schema 和 Parquet schema 的大小写问题
https://blog.csdn.net/MrZhangBaby/article/details/133793660


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

## SQL转化为MapReduce的过程
Hive SQL背后的原理： https://tech.meituan.com/2014/02/12/hive-sql-to-mapreduce.html

* Antlr定义SQL的语法规则，完成SQL词法，语法解析，将SQL转化为抽象语法树AST Tree
* 遍历AST Tree，抽象出查询的基本组成单元 QueryBlock
* 遍历QueryBlock，翻译为执行操作树 OperatorTree
* 逻辑层优化器进行OperatorTree变换，合并不必要的ReduceSinkOperator，减少shuffle数据量
* 遍历OperatorTree，翻译为MapReduce任务
* 物理层优化器进行MapReduce任务的变换，生成最终的执行计划


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



## pig语法
https://pig.apache.org/docs/latest/basic.html

* group XXX by yyy
  * 得到一个inner bag， 以大括号框起来
* limit a
  * 截断输出tuples的a个元素
  * 如果limit的个数a比tuples总个数少，那么输出的内容是不确定的。 There is no guarantee which n tuples will be returned, and the tuples that are returned can change from one run to the next. 除非你使用order操作, A particular set of tuples can be requested using the ORDER operator followed by LIMIT.



# PrestoSQL

presto是Java实现的， 由Facebook开源，为了解决hive查询慢产生。提供的用户语言也是SQL。

## trino 
2020年底，PrestoSQL更名为Trino


# Impala
impala由C++实现，提供的用户语言也是SQL。调用C语言层的libhdfs来直接访问HDFS处理数据。

impala 使用hive的元数据, 完全在内存中计算。 使用上和Presto很接近。


# Spark 
当前最流行的批处理计算引擎应该就是spark了。

* spark-submit 提交jar包运行
  * --master 
    * yarn : in YARN mode the ResourceManager’s address is picked up from the Hadoop configuration.
    * spark://host:port
    * k8s://https://host:port
    * local: 默认
  * --deploy-mode
    * In cluster mode, the Spark driver runs inside an application master process which is managed by YARN on the cluster, and the client can go away after initiating the application. 
    * In client mode, the driver runs in the client process, and the application master is only used for requesting resources from YARN. 默认
  * conf配置
  * --num-executors 50 \ executors的数量
  * --executor-memory 4G \ 每个executor的内存
  * --executor-cores 2 \ 每个executor的核数
  * --driver-memory 2G \ driver的内存
  * --conf spark.storage.memoryFraction=0.6 \ 用于缓存的内存占比
  * --conf spark.shuffle.memoryFraction=0.2 \ 用于shuffle的内存占比
  * --conf spark.locality.wait=10s \ task在executor中执行之前的等待时间
  * --conf spark.shuffle.file.buffer=64k \ shuffle过程中读取文件的缓冲区大小
  * --conf spark.yarn.executor.memoryOverhead=2048 \ 设置堆外内存， 2.3之前版本
  * --conf spark.executor.memoryOverhead=2048 \ 对外内存， 2.3之后版本
  * --conf "spark.memory.offHeap.enabled=true" \
  * --conf "spark.memory.offHeap.size=4096mb" \
  * --conf spark.core.connection.ack.wait.timeout=300 \ ack超时时间
  * --conf spark.network.timeout=120s \ 网络超时时间
  * --conf spark.default.parallelism=800 \
* spark-shell 命令行交互开发
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
* mapPartitionsFunction

## sql/java/scala的类型兼容性，并不是很友好
* java.lang.ClassCastException: scala.collection.mutable.WrappedArray$ofRef cannot be cast to java.util.List
* java.lang.RuntimeException: java.util.HashMap is not a valid external type for schema of map
* java.lang.RuntimeException: java.util.Collections$UnmodifiableRandomAccessList is not a valid external type for schema of array
* java.lang.RuntimeException: java.util.ArrayList is not a valid external type for schema of array


## PySpark

pyspark给特别喜欢写python脚本语言而不喜欢写java的人提供了机会；对于简单的处理逻辑，脚本无需编译直接运行，很方便。
如果你的运行逻辑需要依赖三方库，在pyspark库中引入复杂的第三方库，可没有pip或conda那么简单了，需要做新的spark集群镜像。

## SparkSQL
SparkSQL引擎
Spark SQL can use existing Hive metastores, SerDes, and UDFs.

> spark-sql

```java
    Dataset<Row> df = sparkSession.sql(cmdLine.getSql());
```


```shell
echo  "show databases;" > test.sql
spark-sql -f  test.sql
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