---
title: "使用机器学习做自然语言处理"
subtitle: "NLP using Machine Learning"
categories: [design]
layout: post
---

自然语言处理和信息检索有一定的关系，因为搜索的查询词本身是自然语言，对查询词有合适的理解和表达，能够帮助搜索系统更好的工作。

# Terminology

## Tag

### BMES Tag
标签为B（Begin）、M（Middle）、E（End）、S（Single）

### BB2B3MES Tag





## Named-entity recognition

### Inside–outside–beginning tagging


## 词性标注 Part-of-speech tagging
https://web.stanford.edu/class/cs124/lec/postagging.pdf



## N-gram

An n-gram is a contiguous sequence of n items from a given sample of text or speech.

### Unigram

A bigram is an n-gram for n=1.

### Bigram

A bigram is an n-gram for n=2.


## Word Embedding

word2vec model
这不是一个特定的模型，而是一类模型。
word2vec is a group of related models that are used to produce word embeddings.
模型的输入是一段语料。输出是词向量空间。

具体到算法，我们这么把一个词变为一个浮点数向量？

Skip-Gram model

Bag of words model

Continuous Bag of Words（CBOW）model







# Problem

## Sequence Tagging Problem

CRF, HMM, MEMM

CRF(条件随机场)，HMM(隐马模型)，MEMM(最大熵隐马模型)都常用来做序列标注的建模，像分词、词性标注，以及命名实体标注。
隐马模型一个最大的缺点就是由于其输出独立性假设，导致其不能考虑上下文的特征，限制了特征的选择。
最大熵隐马模型则解决了隐马的问题，可以任意选择特征，但由于其在每一节点都要进行归一化，所以只能找到局部的最优值，同时也带来了标记偏见的问题，即凡是训练语料中未出现的情况全都忽略掉。
条件随机场则很好的解决了这一问题，他并不在每一个节点进行归一化，而是所有特征进行全局归一化，因此可以求得全局的最优值。



## Sequence Classification Problem

给一个文本序列一个分类标签。


Recurrent Neural Networks

LSTM Networks

Bi-LSTM

self-attention mechanism


## 意图分类

输入为词序列，输出为一个意图。






# Framework

以上讲的算法、理论、模型，都有成熟的开源项目来实现。

## word2vec

上面说了 word2vec 是用来产生词向量的模型。但其实有一个同名的开源项目来实现了这个模型。
https://code.google.com/archive/p/word2vec/
这个工具把一个文本语录集合作为输入，输出word vector。

demo scripts: ./demo-word.sh and ./demo-phrases.sh

可以试试中文预料，当然要先分词。http://www.cnblogs.com/hebin/p/3507609.html

关键代码：
  word2phrase
  word2vec
  distance


## CRF++



## fasttext

This library has two main use cases: word representation learning and text classification.

$ ./fasttext skipgram -input data.txt -output model


## Gensim

这是一个Python库，用来做语义相似度。
这个库实现了tf-idf, random projections, word2vec and document2vec algorithms, hierarchical Dirichlet processes (HDP), latent semantic analysis (LSA, LSI, SVD) and latent Dirichlet allocation (LDA).




## Tensorflow

Tensorflow属于大而全的功能框架，我有另一篇[Blog文章](https://lucky521.github.io/blog/design/2017/10/26/tensorflow.html)里单独描述的。




# Reference

Sequence Tagging with Tensorflow https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html

斯坦福的cs224n http://web.stanford.edu/class/cs224n/


https://www.tensorflow.org/tutorials/word2vec
