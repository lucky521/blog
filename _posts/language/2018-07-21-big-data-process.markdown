---
title: "大数据处理的方法"
subtitle: "Working on Big Data"
categories: [framework]
layout: post
---

本篇想要总结一下在Hadoop集群上处理大数据文件的方法，主要从应用层面去看我们有多少种、怎么样的方式去读写大数据进行增删查改，不打算关注框架的实现部分。


# Hive



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




# Spark

## PySpark


# Flink


