---
title: "使用jupyter notebook工作学习"
subtitle: ""
categories: [productivity]
layout: post
---

jupyter使得代码、文档、可视化可以整合到一起，交互式开发环境。


# This is jupyter notebook

ipynb文件既是笔记、又是代码，这对于边学边练的方式非常非常有用。 

Jupyter notebook, 前身是IPython notebook。

jupyter notebook 将python和markdown结合在同一个文本文件。ipynb文件在呈现的时候既会按markdown格式呈现出文字段落、python代码，同时又会按顺序试试执行代码，将print的结果输出出来。


```python
import sys
print sys.version
```

    2.7.10 (default, Feb  7 2017, 00:08:15)
    [GCC 4.2.1 Compatible Apple LLVM 8.0.0 (clang-800.0.34)]



暂时我还没有办法在github jekyll 直接解析ipynb文件。而需要事先使用命令将ipynb转换为markdown文件。

  $ jupyter nbconvert hello.ipynb --to markdown


# JupyterLab 快捷技巧

使用 jupyterLab 的目的是尽可能的把各种开发工具高效的结合在一个平台上。所以高效是一个重点。

在工作场景下，JupyterLab 和 Docker结合起来，构成了方便而强大了工作平台。

## 核心功能

和本地的快速数据传递

Python开发环境

大数据集群环境

分布式机器学习训练平台

各种机器的快速跳板


## 快捷键

突然离线的时候

持续的远端操作环境


# 参考

- 用Jupyter可以做什么事情？ https://www.zhihu.com/question/46309360