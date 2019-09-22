---
title: "C++编码风格"
categories: [Language]
layout: post
---


# cpplint


# 指针

shared_ptr, weak_ptr, unique_ptr

unique_ptr是用于取代c++98的auto_ptr的产物



# const



# 匿名函数


# std::forward


# std::move

std::move(t) 用来表明对象t 是可以moved from的,它允许高效的从t资源转换到lvalue上.



# 并行安全的数据结构


# Lock

std::unique_lock<std::mutex>

std::lock_guard<std::mutex>