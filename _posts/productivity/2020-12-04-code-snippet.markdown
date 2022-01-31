---
title: "经常反复使用的代码片段"
subtitle: ""
categories: [productivity]
layout: post
---


## Python

### unicode转中文
```python
"\xe6\x82".encode('latin-1').decode('utf-8')
```

## 本地文件Shell处理

### 删除本地老数据
find ./-type f -mtime +30 -exec rm -f {} \; 


### 并行执行（fork）

```shell
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


### 删除一天之前的数据
hadoop fs -ls hdfs://ns1012/xxx/xxxx/*    |   tr -s " "    |    cut -d' ' -f6-8    |     grep "^[0-9]"    |    awk 'BEGIN{ MIN=1440; LAST=60*MIN; "date +%s" | getline NOW } { cmd="date -d'\''"$1" "$2"'\'' +%s"; cmd | getline WHEN; DIFF=NOW-WHEN; if(DIFF > LAST){ print "Deleting: "$3; system("hadoop fs -rm -r "$3) }}'


## Hive

### 获取hive表最新分区
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

### 删除Hive表的特定分区

dt=`date -d "$date_base -2 days" "+%Y-%m-%d"`
`hive -e "use search; alter table  XXXXXXXXX  drop partition(dt='$dt_delete')"`



## Java

### 浮点数显示
任何浮点数，当设计显示，一定就是以string来显示的。
```java
import java.text.DecimalFormat;

public class Test {
    private static String try_format(double d) {
      return new DecimalFormat("0.00000000000000").format(d).replaceAll("(?<!\\.)[0]*$","");
    }
    
    private static void try_print(double d) {
        System.out.println(try_format(d));
        System.out.println(d);
        System.out.println("\n");
    }
    
    public static void main(String[] args) {
        liulu_print(-0.000000005454);
        liulu_print(34534534.0);
        liulu_print(0.50);
        liulu_print(0.0);
        liulu_print(-1.0);
        liulu_print(-0.0843503);
        liulu_print(-0.0423040);
    }
}
```