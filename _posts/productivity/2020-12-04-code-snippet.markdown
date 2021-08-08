---
title: "经常反复使用的代码片段"
subtitle: ""
categories: [productivity]
layout: post
---



## 本地文件

### 删除本地老数据
find ./-type f -mtime +30 -exec rm -f {} \; 


### 并行执行（fork）

```
N=4
(
for thing in a b c d e f g h i g j k l m; do 
   echo "i0=${i}"; ((i=i%N)); ((i++==0)) && echo "i1=${i}" && wait ; (echo "I am doing $thing i2=${i}"; sleep 1 ) &
done
)
```


## HDFS

### 删除hdfs老数据
hadoop fs -rm -r `hadoop fs -ls  /user/recsys/rank/arch/checkpoint/sample_join/ | grep ' 2020-09.*' | awk '{print $8}'`


## Hive

### 获取最新分区
```shell
hive -e "set hive.cli.print.header=false;show partitions app.app_query_attr_feature;" | tail -1 | cut -d'=' -f2
```



### 修复hive 表分区查询
 set hive.msck.path.validation=ignore; MSCK REPAIR TABLE 表名




### Hive 灌入测试数据
INSERT overwrite table rnn_user_state_embedding
PARTITION (dt='2020-11-10')
select 'id001', array(cast(1.0 as float),cast(2.0 as float)) 
from touchstone_info limit 1


### Hive删除表
 DROP TABLE IF EXISTS rnn_user_state_embedding