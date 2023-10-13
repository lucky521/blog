---
title: "一个编程语言的基本要素"
categories: [Language]
layout: post
---

# 编程语言的功能

我们来看看一门完善的编程语言中需要提供什么内容给程序开发者。

## 语法结构

## 数据类型支持

## 动态内存管理支持

函数 | 作用
----|----
malloc | 分配内存
calloc | 分配内存并初始化为0
realloc | 扩展之前申请过的内存块
free | 回收空间
aligned_alloc | 分类对齐的内存块

## 输入输出支持

## 数学和算法支持

## 原子操作支持

## 并发支持





# 编程语言的解析

编程语言是如何提供以上支持的功能？

任何的规则化的文本数据，都可以实现解析器来解析。

## BNF

BNF是John Backus 在20世纪90年代提出的用以简洁描述一种编程语言的语言。

## lex & yacc

注意这不仅仅是用于编译语言的，更多适合用于文本处理。

bison/flex


https://github.com/Engelberg/instaparse

## antlr4

Hive/Spark的SQL解析是使用的antlr4


## zetasql

# 编译器后端

指令选择，指令调度，寄存器分配