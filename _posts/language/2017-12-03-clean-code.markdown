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

## HiveSQL

https://ytluck.github.io/data-mining/my-dataming-post-42.html

  

### 优化选项

* 选取执行引擎： mr tez spark  , set hive execution.engine=spark
* 选取文件存储格式:
* 选取优化器: set  hive.cbo.enable=true

* 调整map数
```
    map读取文件的大小： set  dfs.block.size
    map 的个数   set mapred.min.split.size
            set mapred.max.split.size
```

* 合并小文件
```
    hive.merg.mapfiles=true：合并map输出 
    hive.merge.mapredfiles=false：合并reduce输出 
    hive.merge.size.per.task=256*1000*1000：合并文件的大小 
    hive.mergejob.maponly=true：如果支持CombineHiveInputFormat则生成只有Map的任务执行merge 
    hive.merge.smallfiles.avgsize=16000000：文件的平均大小小于该值时，会启动一个MR任务执行merge。
```





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



