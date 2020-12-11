---
title: "高质量代码"
categories: [Language]
layout: post
---

# 高质量代码的特征

没bug

好理解

好扩展

性能好

好复用

区别高质量的代码和高质量的设计



# 高质量代码的例子

```cpp

```

```java

```

```python

```

```shell

```



# 高质量代码的原则

SLAP 抽象层次一致性原则



# 分语言细说原则

## C/C++

## Java

## SQL/Hive

* 存储符合 ISO-8601 标准的日期格式（YYYY-MM-DD HH:MM:SS.SSSSS）
* 在 SQL 代码中加入注释。优先使用 C 语言式的以 /* 开始以 */ 结束的块注释，或使用以 -- 开始的行注释，并在末尾换行。
* 关键字总是大写，如 SELECT 和 WHERE。
* 只select 需要的列，而不select *， 减少无用的数据传输。
* 使用 explain 关键字来查看SQL的执行计划。
* 怎么样存储数据？ 使用列式存储： 适合不需要频繁删除/更新数据的表。独立存储，且数据类型已知，可以针对该列的数据类型、数据量大小等因素动态选择压缩算法，以提高物理存储利用率。可在数据列中高效查找数据，无需维护索引(任何列都能作为索引)，查询过程中能够尽量减少无关IO，避免全表扫描。
* 怎么样压缩数据？
* 怎么样采样数据？ 不要用ORDER BY RAND()，建议自己实现根据某字段的hash取模分桶。
* 怎么样排序数据？ ORDER BY是全局排序，SORT BY会先局部排序再归并，DISTRIBUTE BY控制map的输出在reducer中是如何划分的。
* 怎么样join数据？ 数据量少的表放左边，在 Join 操作的 Reduce 阶段，位于 Join 操作符左边的表的内容会被加载进内存（在join操作的每一个mapred程序中，hive都会把出现在join语句中相对靠后的表的数据stream化，相对靠前的变的数据缓存在内存中）。在进行 join 操作的条件过滤的时候，应该将过滤条件放在 on 关键词里面，提高查询的效率。由于join操作是在where操作之前执行，所以当你在执行join时，where条件并不能起到减少join数据的作用。
* 怎么去重数据？ 尽量使用group by替代distinct。
* 避免数据倾斜：无效key导致的倾斜；不同数据类型join产生的倾斜；
* 怎么发现数据倾斜？reduce卡到99%不动或者某几个reduce长时间的执行


### 优化选项

* 合理的mapper数量
job会通过input文件产生一个或者多个map数，主要的决定因素是input文件数和input文件大小。
mapper过多会有更多初始化和创建开销，产出小文件也过多；mapper过少并发度会不够。

* 合理的reducer数量
reducer过多会使得最终输出的小文件多，影响下游；reducer过少每个文件会很大易OOM，且容易数据倾斜。

```
hive.exec.reducers.bytes.per.reducer ＃这个参数控制一个job会有多少个reducer来处理，依据的是输入文件的总大小。默认1GB。
hive.exec.reducers.max ＃这个参数控制最大的reducer的数量， 如果 input / bytes per reduce > max 则会启动这个参数所指定的reduce个数。 这个并不会影响mapre.reduce.tasks参数的设置。默认的max是999。
mapred.reduce.tasks ＃这个参数如果指定了，hive就不会用它的estimation函数来自动计算reduce的个数，而是用这个参数来启动reducer。默认是-1。
```

* 合并小文件
```
// 是否和并 Map 输出文件，默认为 True
hive.merge.mapfiles = true
// 是否合并 Reduce 输出文件，默认为 False
hive.merge.mapredfiles = false
// 合并文件的大小
hive.merge.size.per.task = 256*1000*1000
```



set hive.exec.parallel=true;
set hive.exec.compress.output=true
set hive.exec.compress.intermediate=true
set mapreduce.map.memory
set mapreduce.reduce
set hive.exec.mode.local.auto



https://www.jianshu.com/p/6970c47eec5c

https://developer.aliyun.com/article/741117

https://blog.csdn.net/scgaliguodong123_/article/details/45477323

https://www.cnblogs.com/frankdeng/p/9463897.html

## Shell

## Python

## Javascript




# 维护高质量代码的途径/工具

仅靠人的高度自觉和高水平是无法维护高质量代码的，软件工程是劳动密集型工业。

* 强制静态检查（cpplint，语法、对齐、命名

* CodeReview利益 （靠人的主动是不行的，需要利益来支持

* 强制自动化测试

* 强制单元测试

* 持续集成




# Trade-off

## "临时方案"

“这样着急，先这样，下次一定”。

## 代码质量和功能时效的trade-off

代码是服务于功能和产品的，抛开后者说问题，就会过于理想主义，而实际上，许多情况下，我们没有能力或者必要极度的高质量，产品功能尽快实现比质量更为重要。

## 代码质量和人力资源

维持高的代码的质量，短期拖慢进度？长期加快进度？

## 在线代码和离线代码

* 在线代码指的是程序交付给用户或者用户使用时会直接运行到的代码，比如线上搜索服务、客户端App等，这部分如果有问题，容易使得用户直接感知到，甚至难以及时修复，比如程序coredump，服务部响应等。
* 离线代码指的是不会直接交付给用户或者用户使用时也不会直接运行到的代码，比如模型训练代码、大数据处理任务等，这部分如果有问题，一般不会直接将传达到用户一侧，或者没有直接的影响。

以上的划分不能说明两类代码的重要程度，但解释了一种现象，离线代码的质量相比在线代码更容易被忽视。



