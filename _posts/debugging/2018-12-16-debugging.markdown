---
title: "什么是Debugging"
categories: [debugging]
layout: post
---

在维基百科上，Debugging的定义是：Debugging is the process of finding and resolving defects or problems within a computer program that prevent correct operation of computer software or a system.

简单来讲，就是找问题，查问题。


# 代码 Debugging

## Windows

WinDgb

## Linux

```shell
gdb binary corefile
```

## Mac

LLDB Debugger

## Java 


HotSpot VM的Native Memory Tracking

java8给HotSpot VM引入了Native Memory Tracking (NMT)特性，可以用于追踪JVM的内部内存使用

使用-XX:NativeMemoryTracking=summary可以用于开启NMT，其中该值默认为off，可以设置summary、detail来开启；开启的话，大概会增加5%-10%的性能消耗；使用-XX:+UnlockDiagnosticVMOptions -XX:+PrintNMTStatistics可以在jvm shutdown的时候输出整体的native memory统计；其他的可以使用jcmd pid VM.native_memory相关命令进行查看、diff、shutdown等



## Static Code Analysis

Lint

静态程序分析工具




# 状态 Debugging

http://www.brendangregg.com/linuxperf.html



# 性能 Debugging

如何高效的衡量程序性能？

http://www.brendangregg.com/linuxperf.html


火焰图 Flame Graph
https://github.com/brendangregg/FlameGraph


## 对一个机器进行快速体检

http://www.brendangregg.com/Articles/Netflix_Linux_Perf_Analysis_60s.pdf

10个命令，对一个机器的负载情况进行快速体检
* uptime 指明过去1、5、15分钟CPU负载过的进程个数
* dmesg | tail



# Reference

https://en.wikipedia.org/wiki/Debugging

[Brendan Gregg](http://www.brendangregg.com/)

[最全的Java服务问题排查套路](https://mp.weixin.qq.com/s?__biz=MzUzODQ0MDY2Nw==&mid=2247483975&idx=1&sn=14dad1cf52a4407456eaf32395902bb7)