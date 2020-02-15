---
title: "C++编码风格"
categories: [Language]
layout: post
---


# cpplint


# 智能指针

shared_ptr, weak_ptr, unique_ptr

unique_ptr是用于取代c++98的auto_ptr的产物



# const


# 仿函数

# 匿名函数


# 左值右值

C++中的每个表达式都有两种独立的特性：类型（type）和值分类（value category）。每个表达式都属于三大value category之一：prvalue，xvalue和lvalue。

# 右值引用

# std::forward

完美转发实现了参数在传递过程中保持其值属性的功能，即若是左值，则传递之后仍然是左值，若是右值，则传递之后仍然是右值。

std::forward<T>()不仅可以保持左值或者右值不变，同时还可以保持const、Lreference、Rreference、validate等属性不变；


# std::move

std::move(t) 用来表明对象t 是可以moved from的,它允许高效的从t资源转换到lvalue上.



# 并行安全的数据结构


# Lock

std::unique_lock<std::mutex>

std::lock_guard<std::mutex>