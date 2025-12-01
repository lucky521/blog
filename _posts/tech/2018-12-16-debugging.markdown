---
title: "什么是Debugging"
categories: [Tech]
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


## coredump分析

核心转储（core dump）是程序在遇到严重错误时由操作系统生成的，用于保存程序异常终止时的内存镜像。这种机制主要用于后续的调试和分析，以确定程序崩溃的原因。产生核心转储的信号有很多，以下是一些常见的信号：

### 1. SIGSEGV (信号 11)
- **描述**：段错误（Segmentation Fault），当程序尝试访问其内存地址空间外的内存区域时发生。
- **常见原因**：空指针解引用、数组越界、访问已释放的内存。

### 2. SIGABRT (信号 6)
- **描述**：异常终止（Abort Signal），通常是因为`abort()`函数被调用。
- **常见原因**：程序自检测到一个严重的错误条件并决定终止。

### 3. SIGFPE (信号 8)
- **描述**：浮点异常（Floating-Point Exception），在发生致命的算术运算错误时发生。
- **常见原因**：除以零、溢出等算术错误。

### 4. SIGILL (信号 4)
- **描述**：非法指令（Illegal Instruction），当程序尝试执行一条非法或未定义的机器语言指令时发生。
- **常见原因**：程序二进制文件损坏、尝试执行非法的机器指令。

### 5. SIGBUS (信号 7)
- **描述**：总线错误（Bus Error），当程序的内存访问不符合硬件系统的物理地址要求时发生。
- **常见原因**：内存对齐错误，尝试访问一个不能正确对齐在硬件上的内存地址。

### 6. SIGQUIT (信号 3)
- **描述**：退出信号（Quit Signal），通常由用户发送，用于请求程序终止并生成核心转储。
- **常见原因**：用户希望终止程序并生成核心转储以进行调试。

### 7. SIGTRAP (信号 5)
- **描述**：跟踪/断点陷阱（Trace/Breakpoint Trap）。
- **常见原因**：调试过程中遇到断点或其他跟踪条件。

这些信号中的每一个都指示了程序运行中遇到的特定类型的错误。理解这些信号及其背后的原因对于调试和解决问题至关重要。使用调试器（如GDB）可以帮助开发者更精确地定位问题的源头，并分析程序崩溃的原因。



# 状态 Debugging

http://www.brendangregg.com/linuxperf.html

哪些状态是需要关注的？



# 性能 Debugging

[如何高效的衡量程序性能？](http://www.brendangregg.com/linuxperf.html)

[火焰图 Flame Graph](https://github.com/brendangregg/FlameGraph)

```sh
sudo yum install perf
perf record -F 99 -p 进程号 -g -- sleep 秒数
perf script > out.perf

perf report -i perf.data
perf top -e cpu-clock -p 进程号

git clone https://github.com/brendangregg/FlameGraph.git
cd FlameGraph
./stackcollapse-perf.pl out.perf > out.folded
./flamegraph.pl out.folded > out.svg

```


## 对一个机器进行快速体检

http://www.brendangregg.com/Articles/Netflix_Linux_Perf_Analysis_60s.pdf

10个命令，对一个机器的负载情况进行快速体检
* uptime 指明过去1、5、15分钟CPU负载过的进程个数
* dmesg | tail



# Reference

https://en.wikipedia.org/wiki/Debugging

[Brendan Gregg](http://www.brendangregg.com/)

[最全的Java服务问题排查套路](https://mp.weixin.qq.com/s?__biz=MzUzODQ0MDY2Nw==&mid=2247483975&idx=1&sn=14dad1cf52a4407456eaf32395902bb7)