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