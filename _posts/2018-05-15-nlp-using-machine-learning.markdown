---
title: "使用机器学习做自然语言处理"
subtitle: "NLP using Machine Learning"
categories: [design]
layout: post
---

自然语言处理和信息检索有一定的关系，因为搜索的查询词本身是自然语言，对查询词有合适的理解和表达，能够帮助搜索系统更好的工作。

# Concept

## Word Embedding

word2vec model
这不是一个特定的模型，而是一类模型。word2vec is a group of related models that are used to produce word embeddings.
模型的输入是一段语料。输出是词向量空间。

Skip-Gram model

Bag of words model

连续词袋模型（CBOW）模型







# Problem

## Sequence Tagging Problem

CRF


## Sequence Classification Problem

给一个文本序列一个分类标签。


Recurrent Neural Networks

LSTM Networks

Bi-LSTM

self-attention mechanism


### 意图分类

输入为词序列，输出为一个意图。



# Framework

以上讲的算法、理论、模型，都有成熟的开源项目来实现。

## word2vec

上面说了 word2vec 是用来产生词向量的模型。但其实有一个同名的开源项目来实现了这个模型。
https://code.google.com/archive/p/word2vec/
这个工具把一个文本语录集合作为输入，输出word vector。

demo scripts: ./demo-word.sh and ./demo-phrases.sh

可以试试中文预料，当然要先分词。http://www.cnblogs.com/hebin/p/3507609.html


## fasttext

This library has two main use cases: word representation learning and text classification.


## Gensim

这是一个Python库，用来做语义相似度。
这个库实现了tf-idf, random projections, word2vec and document2vec algorithms, hierarchical Dirichlet processes (HDP), latent semantic analysis (LSA, LSI, SVD) and latent Dirichlet allocation (LDA).


## Tensorflow

Tensorflow属于大而全的功能框架，我有另一篇[Blog文章](https://lucky521.github.io/blog/design/2017/10/26/tensorflow.html)里单独描述的。


# Reference

Sequence Tagging with Tensorflow https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html

斯坦福的cs224n http://web.stanford.edu/class/cs224n/


https://www.tensorflow.org/tutorials/word2vec
