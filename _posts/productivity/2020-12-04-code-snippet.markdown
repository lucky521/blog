---
title: "经常反复使用的代码片段"
subtitle: ""
categories: [productivity]
layout: post
---


## linux命令

```shell
 curl -d 'username=lu.dev' -d 'tableName=db.tablename' -X POST  http://xxx.com/api/v1/table/detail
```

```shell
ps -eo pid,lstart,etime,cmd | grep bash | grep 2019 | awk '{print $1}' | xargs kill -9
```


```shell
#!/bin/bash
function cpu(){
    
    util=$(vmstat | awk '{if(NR==3)print $13+$14}')
    iowait=$(vmstat | awk '{if(NR==3)print $16}')
    echo "CPU -使用率：${util}% ,等待磁盘IO相应使用率：${iowait}:${iowait}%"
 
}
function memory (){
 
    total=`free -m |awk '{if(NR==2)printf "%.1f",$2/1024}'`
    used=`free -m |awk '{if(NR==2) printf "%.1f",($2-$NF)/1024}'`
    available=`free -m |awk '{if(NR==2) printf "%.1f",$NF/1024}'`
    echo "内存 - 总大小: ${total}G , 使用: ${used}G , 剩余: ${available}G"
}
disk(){
    
    fs=$(df -h |awk '/^\/dev/{print $1}')
    for p in $fs; do
        mounted=$(df -h |awk '$1=="'$p'"{print $NF}')
        size=$(df -h |awk '$1=="'$p'"{print $2}')
        used=$(df -h |awk '$1=="'$p'"{print $3}')
        used_percent=$(df -h |awk '$1=="'$p'"{print $5}')
        echo "硬盘 - 挂载点: $mounted , 总大小: $size , 使用: $used , 使用率: $used_percent"
    done
 
}
function tcp_status() {
    summary=$(ss -antp |awk '{status[$1]++}END{for(i in status) printf i":"status[i]" "}')
    echo "TCP连接状态 - $summary"
}
cpu
memory
disk
tcp_status
```



## 本地Shell

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

### 去除字符串左侧的0

hour="01" &&  hour_int=$(echo $hour | sed 's/^0//') && echo $hour_int

## HDFS

### 删除hdfs老数据
hadoop fs -rm -r `hadoop fs -ls  /user/recsys/rank/arch/checkpoint/sample_join/ | grep ' 2020-09.*' | awk '{print $8}'`


### 删除一天之前的数据
hadoop fs -ls hdfs://ns1012/xxx/xxxx/* | tr -s " " | cut -d' ' -f6-8 | grep "^[0-9]" | awk 'BEGIN{ MIN=1440; LAST=60*MIN; "date +%s" | getline NOW } { cmd="date -d'\''"$1" "$2"'\'' +%s"; cmd | getline WHEN; DIFF=NOW-WHEN; if(DIFF > LAST){ print "Deleting: "$3; system("hadoop fs -rm -r "$3) }}'


### 删除hive表元信息中不再需要的分区
老旧的hive版本不支持在repair的时候自动删除不存在的分区。
```shell
# sh clean_meta.sh 2021-12-29 00 unknown xxxx
dt=$1
dh=$2
channel=$3
topic=$4
echo "Checking dt=$dt,dh=$dh,channel=$channel,topic=$topic"
hadoop fs -ls hdfs://xxx/xxx/my.db/mytable/dt=$dt/dh=$dh/channel=$channel/topic=$topic
if [ $? -ne 0 ]; then
    echo "file not exists. Drop meta of dt=$dt,dh=$dh,channel=$channel,topic=$topic"    
    hive -e "ALTER TABLE my.mytable DROP IF EXISTS PARTITION(dt='$dt',dh='$dh',channel='$channel',topic='$topic');"
fi
```


## SQL


### 获取一个表的建表语句
```
SHOW CREATE TABLE table_name;
```

### 查询特定的分区
```shell
show partitions tmpr.rec_feature_log_data partition(dt='2022-06-20',pid='660000');
```

### 按条件范围删除旧分区
```shell
alter table search.search_rank_feature_log_v1_remote_parquet drop partition (dt<'2022-01-01',topic='jxpp')
```


### 获取hive表最新分区
```shell
hive -e "set hive.cli.print.header=false;show partitions app.app_query_attr_feature;" | tail -1 | cut -d'=' -f2
```

### 修改hive表的元信息
 ALTER TABLE xxx SET TBLPROPERTIES('comment' = 'XXX');

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


### 过滤空字段

https://stackoverflow.com/questions/18432925/handling-null-values-in-hive

### 控制MR工作量分配的参数
如果输入原始小文件过多，容易出现map数过多，每个map处理时间太短的时间，大量开销花费在资源分配上了。

hive_para="
    set mapreduce.map.memory.mb=5120;
    set mapreduce.map.java.opts=-Xmx4096M;
    set mapreduce.map.cpu.vcores = 4;
    set mapreduce.job.reduce.slowstart.completedmaps=1;
    set mapreduce.reduce.memory.mb=8192;
    set mapreduce.reduce.java.opts=-Xmx7168M;
    set mapreduce.reduce.cpu.vcores = 8;
    set hive.input.format=org.apache.hadoop.hive.ql.io.CombineHiveInputFormat;
    set hive.hadoop.supports.splittable.combineinputformat = true;
    set mapreduce.input.fileinputformat.split.maxsize = 1073741824;
    set mapreduce.input.fileinputformat.split.minsize.per.node=1073741824;
    set mapreduce.input.fileinputformat.split.minsize.per.rack=1073741824; 
    set hive.exec.reducers.bytes.per.reducer = 2147483648;
    set hive.merge.mapfiles = true;
    set hive.merge.mapredfiles = true;
    set hive.merge.size.per.task = 256000000;
    set hive.merge.smallfiles.avgsize = 256000000;
    set hive.exec.dynamic.partition=true;
    set hive.exec.dynamic.partition.mode=nonstrict;
    set hive.exec.max.dynamic.partitions.pernode = 1000;
    set hive.exec.max.dynamic.partitions=1000;
    set hive.optimize.skewjoin=true;
    set hive.skewjoin.key=100000;
    set hive.map.aggr=true;
    set hive.groupby.mapaggr.checkinterval = 100000;
    set hive.map.aggr.hash.min.reduction=0.5;
    set hive.exec.max.created.files=2000000;
    set mapreduce.job.split.metainfo.maxsize=-1;
    set mapreduce.map.speculative=false;
    set mapreduce.reduce.speculative=false;
"

### 怎么join最高效

left semi join 


## Python

### unicode转中文
```python
"\xe8\xb7\xaf\xe7\x94\xb1\xe6\x89\xbe\xe4\xb8\x8d\xe5\x88\xb0\xe5\x95\xa6\xef\xbc\x81".encode('latin-1').decode('utf-8')
```


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

### 创建线程池
```java
import java.util.concurrent.Executor;
创建一个可重用固定线程数的线程池，以共享的无界队列方式来运行这些线程。

```

### 创建子进程
https://zetcode.com/java/processbuilder/
```java
    ProcessBuilder pb = new ProcessBuilder("myCommand", "myArg1", "myArg2");
    Map<String, String> env = pb.environment();  // 环境变量 
    env.put("VAR1", "myValue");
    env.remove("OTHERVAR");
    env.put("VAR2", env.get("VAR1") + "suffix");
    pb.directory(new File("myDir"));  // 文件
    File log = new File("log");
    pb.redirectErrorStream(true);
    pb.redirectOutput(Redirect.appendTo(log));
    Process p = pb.start();  // 开始子进程
    assert pb.redirectInput() == Redirect.PIPE;
    assert pb.redirectOutput().file() == log;
    assert p.getInputStream().read() == -1;
```




# 参考大全

https://zetcode.com/