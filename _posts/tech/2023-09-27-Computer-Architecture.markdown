---
title: "计算机体系结构：量化研究方法"
subtitle: "Computer Architecture: A Quantitative Approach"
categories: [Tech]
layout: post
---

这是一本关于计算机系统的处理器、存储器、GPU、并行优化的巨作。
两位作者是图灵奖获得者、斯坦福/UC伯克利教授、Google科学家。

目前（2021年）最新的版本是2019年的第六版，还没有中文翻译版本。上一版第五版是2012年出版，已经有中文版。不过还是推荐阅读最新版本。



# 第一章 量化设计和分析基础
2004年开始Intel将重心从研发高性能单处理器转移到多处理器芯片上，不再单独依赖ILP指令集并行，而将更多使用DLP数据集并行，TLP线程级并行。本书第五版引入WSC和RLP请求级并行。
ILP是软件程序不感知的并行，而DLP、TLP、RLP都需要程序显式感知，这也增大了软件编程的难度。

ISA(Instruction Set Architecture)指令集架构
ISA指的是对编程可见的指令集，它作为软件和硬件的交界。如80x86、ARMv8、RISC-V。
RISC-V指令集由UC伯克利研发。具有32个通用寄存器(x0-x31)：
常量0
返回地址
栈指针
全局指针
线程指针
临时
具有32个浮点寄存器(f0-f31)，可以用于32位单精度浮点或64位位双精度浮点。

计算机体系结构
真正的体系结构是有ISA指令集结构、Microarchitecture微架构、硬件三部分组合而成的。
影响体系结构设计的5方面技术：
集成电路、晶体管
半导体DRAM
半导体Flash
磁盘存储
网络技术
摩尔定律:
相同面积的集成电路里，晶体管数目每18-24个月会增加一倍。
阿姆达尔定律：
系统中某一部件因为采用更快的实现后，整个系统性能的提高与该部分的使用频率或者在总运行时间中比例有关。
Dennard缩放定律：
随着晶体管尺寸越来越小，功耗密度保持不变，即单位面积的功耗保持不变。

# 第二章 内存层级设计
CPU寄存器
L1 Cache
L2 Cache
L3 Cache
主内存
闪存或磁盘

Cache评价指标
hit time 从cache将命中数据返回的时间
miss rate 发生cache miss(与cache hit对应)的请求占比
miss penalty 从主内存读取数据并返回的时间开销
cache bandwith  
power consumption

内存延时指标：Access Time 从请求发出到得到响应的时间；Cycle Time 两次无关请求之间的最小时长。
优化缓存性能的十个方法

虚拟存储



第三章 指令级并行
第四章 数据级并行
第五章 线程级并行
第六章 Warehouse-scale computer
（持续更新ing）


