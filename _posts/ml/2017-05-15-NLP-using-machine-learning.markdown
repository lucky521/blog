---
title: "使用机器学习做自然语言处理"
subtitle: "NLP using Machine Learning"
categories: [MachineLearning]
layout: post
---

自然语言处理设计的领域很多，比如机器翻译、问答系统、语音识别、文本分类、中文分词、信息检索、词性标注、知识图谱等等。

自然语言处理和信息检索有一定的关系，因为搜索的查询词本身是自然语言，对查询词有合适的理解和表达，能够帮助搜索系统更好的工作。

语言模型中常见的任务类型: 生成、问答、聊天、 改写、总结、分类等.

中间任务：典型的中间任务包括中文分词、词性标注、NER、句法分析、指代消解、语义Parser等，这类任务一般并不解决应用中的实际需求，大多数是作为那些解决实际需求任务的中间阶段或者辅助阶段存在的。

# NLP Terminology 术语


## token
Token是文本处理中的最小单位，可以是一个单词、一个标点符号或者一个字符。在自然语言处理任务中，通常将文本分割成一系列的Token，用于后续的处理和分析。例如，对于句子 "I love NLP!"，可以将其分割成四个Token：["I", "love", "NLP", "!"]。
### special token
bert中的special token有 [cls],[sep],[unk],[pad],[mask]
[BERT - Tokenization and Encoding](https://albertauyeung.github.io/2020/06/19/bert-tokenization.html/)


## Tag
Tag是对Token的附加信息，用于表示Token的语义、语法或其他属性。标签可以是预定义的，也可以是根据特定任务或应用程序的需求自定义的。标签通常与Token一一对应，用于提供关于Token的更多信息。例如，在词性标注任务中，对于句子 "I love NLP!"，可以为每个Token分配一个词性标签，如["PRON", "VERB", "NOUN", "PUNCT"]，分别表示代词、动词、名词和标点符号。

### BMES Tag
标签为B（Begin）、M（Middle）、E（End）、S（Single）

### BB2B3MES Tag


## Latent Vector Space Models

* LDA
* LSI
* word2vec


## Named-entity recognition

### Inside–Outside–beginning tagging


## 词性标注 Part-of-speech tagging
https://web.stanford.edu/class/cs124/lec/postagging.pdf


## N-gram

将文章以单词级别的划分有时候并不是最好的方法，因为单词的含义与其所处的前后单词关系很大。我们可以将连续出现的N个单词组成一个词组（N-gram），把词组作为特征。

An n-gram is a contiguous sequence of n items from a given sample of text or speech.

### Unigram

A Unigram is an n-gram for n=1.

### Bigram

A bigram is an n-gram for n=2.


## 文本表示模型

一个word的表示。

一段文本的表示。


### 主题模型
主题模型是一种特殊的概率图模型。
把文章表示为K维的主题向量，其中向量的每一维代表一个主题，权重表示这个文章属于该主题的概率。

文章
单词
主题

* pLSA Model
频率派思想。
* LDA Model
贝叶斯派思想



### Word Embedding

word2vec model 这不是一个特定的模型，而是一类模型。该内容在另一篇博客《Data Embedding》中也有体现。
word2vec is a group of related models that are used to produce word embeddings. 

模型的输入是一段语料，输出是词向量空间（real-valued word feature vector）。

具体到算法，我们这么把一个词变为一个浮点数向量？

https://www.cnblogs.com/peghoty/p/3857839.html


### Bag of Words model 词袋模型

bow的表达形式为： bow map < word, number of occurrences > 

不考虑word的顺序，只在意出现次数。

每篇文章可以表示成一个长向量，向量的每一维代表一个单词，每一维的权重反应这个词在文章中的重要程度（TF-IDF）。


### N-gram model

BOW map中的key是一个独立的单词，而N-gram model中，我们把N个单词作为一个key。


### Skip-Gram model

当前词预测上下文：以一个word为输入，试图预测出context word。

skip-gram pair 是要在构造pair的时候把某个单词和其左侧单词pair一次，再和期右侧单词pair一次。

比如"the quick brown fox"被构造成
(quick, the), (quick, brown), (brown, quick), (brown, fox), ...

https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-skip-gram.html

Skip-Gram对低频词更友好，更容易学到低频词的embedding。


### Continuous Bag of Words（CBOW）model

上下文来预测当前词：以一个word的context（周围的word）为输入，试图根据context预测出该word。

cbow pair 每个pair对应位于其中间位置的word。

比如"the quick brown fox jumped over the lazy dog"被构造成
([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...











# NLP Problem 任务目标

## Sequence Tagging / Sequence labelling Problem

CRF, HMM, MEMM

CRF(条件随机场)，HMM(隐马模型)，MEMM(最大熵隐马模型)都常用来做序列标注的建模，像分词、词性标注，以及命名实体标注。
隐马模型一个最大的缺点就是由于其输出独立性假设，导致其不能考虑上下文的特征，限制了特征的选择。
最大熵隐马模型则解决了隐马的问题，可以任意选择特征，但由于其在每一节点都要进行归一化，所以只能找到局部的最优值，同时也带来了标记偏见的问题，即凡是训练语料中未出现的情况全都忽略掉。
条件随机场则很好的解决了这一问题，他并不在每一个节点进行归一化，而是所有特征进行全局归一化，因此可以求得全局的最优值。

序列标注的传统算法总结：http://www.cs.cornell.edu/~nhnguyen/icml07structured.pdf


## Sequence Classification Problem

给一个文本序列一个分类标签。


Recurrent Neural Networks

Bi-LSTM

self-attention mechanism
http://www.jeyzhang.com/understand-attention-in-rnn.html

LSTM with attention 


## 意图分类 Problem

输入为词序列，输出为一个意图。

宽泛理解也可以认为是一种分类问题。






# NLP 典型模型
深度学习之前的NLP模型有以下几种：
* N-gram 模型：
  * N-gram 是一种基于统计的语言模型，用于预测一个词在给定上下文中出现的概率。N-gram 模型基于前 N-1 个词来预测下一个词，其中 N 是一个整数。常见的 N-gram 模型包括 1-gram（也称为 unigram）、2-gram（bigram）和 3-gram（trigram）。
* 隐马尔可夫模型（Hidden Markov Model，HMM）：
  * HMM 是一种基于概率图模型的序列建模方法，用于处理标注和分词等任务。HMM 假设观测序列背后存在一个不可观测的状态序列，并通过状态之间的转移概率和观测概率来建模。
* 最大熵模型（Maximum Entropy Model）：
  * 最大熵模型是一种基于最大熵原理的分类和标注模型。它通过最大化模型的熵来选择最平衡的模型，以满足给定的约束条件。最大熵模型在 NLP 中常用于文本分类和命名实体识别等任务。
* 支持向量机（Support Vector Machines，SVM）：
  * SVM 是一种监督学习算法，用于二分类和多分类任务。在 NLP 中，SVM 可以用于文本分类、情感分析和命名实体识别等任务。通常使用特征工程将文本转换为向量表示，然后使用 SVM 进行分类。
* 条件随机场（Conditional Random Fields，CRF）：
  * CRF 是一种概率图模型，用于序列标注任务。与 HMM 不同，CRF 考虑了输入特征和输出标签之间的依赖关系，并通过条件概率建模。


## Seq2Seq

Seq2Seq模型：预测两个时间序列数据之间的映射关系。

代码：https://github.com/tensorflow/models/tree/r1.4.0/tutorials/rnn/translate


## Attention 机制

在Encoder-Decoder结构中，Encoder把所有的输入序列都编码成一个统一的语义特征c再解码，因此c中必须包含原始序列中的所有信息，它的长度就成了限制模型性能的瓶颈。

attention它是解决 sequence-to-sequence learning 中的这个限制：要求必须把原序列的全部内容压缩到固定长度的vector。

Attention解决这一限制的方法就是：允许decoder回看原序列的 hidden states，这一状态信息作为加权平均值作为decoder的附加输入。

具体计算c_i的方法有很多，比如：我们用 a_{ij} 衡量Encoder中第j阶段的h_j和解码时第i阶段的相关性，最终Decoder中第i阶段的输入的上下文信息 c_i 就来自于所有 h_j 对 a_{ij} 的加权和。

[深度学习中Attention与全连接层的区别何在？](https://www.zhihu.com/question/320174043)

## LSTM Networks

## transformer

Transformer是一种NLP特征抽取器

## Memory 记忆网络








# NLP 开源框架

以上讲的算法、理论、模型，都有成熟的开源项目来实现。

## word2vec

上面说了 word2vec 是用来产生词向量的模型。但其实有一个同名的开源项目来实现了这个模型。
https://code.google.com/archive/p/word2vec/
这个工具把一个文本语录集合作为输入，输出word vector。

由于Softmax激活函数，在迭代过程会遍历词表所有单词，因此迭代过程会非常慢。为此word2vec提出两种加快训练速度的方式，一种是Hierarchical softmax，另一种是Negative Sampling

demo scripts: ./demo-word.sh and ./demo-phrases.sh

可以试试中文预料，当然要先分词。http://www.cnblogs.com/hebin/p/3507609.html

关键代码：
  word2phrase
  word2vec
  distance


## GloVe

斯坦福发布的开源实现 https://github.com/stanfordnlp/GloVe

GloVe is essentially a log-bilinear model with a weighted least-squares objective.


## CRF++
CRF++是著名的条件随机场的开源工具,也是目前综合性能最佳的CRF工具，用于segmenting/labeling sequential data。


## fasttext

This library has two main use cases: word representation learning and text classification.
```shell
$ ./fasttext skipgram -input data.txt -output model
```
官方提供了一份由CBOW模型，输入维基百科语料训练而出的语言模型，每个word为300维向量。
https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md


## Gensim

这是一个Python库，用来做语义相似度。
这个库实现了tf-idf, random projections, word2vec and document2vec algorithms, hierarchical Dirichlet processes (HDP), latent semantic analysis (LSA, LSI, SVD) and latent Dirichlet allocation (LDA).


## Flair
https://github.com/zalandoresearch/flair


## Familia
Familia 开源项目包含文档主题推断工具、语义匹配计算工具以及基于工业级语料


## Tensorflow

Tensorflow属于大而全的功能框架，我有另一篇[Blog文章](https://lucky521.github.io/blog/design/2017/10/26/tensorflow.html)里单独描述的。


## sentencepiece
SentencePiece是一种通用的文本处理工具，用于分割文本成子词或字符级别的单位。它采用基于统计的方法，可以自动学习并生成词汇表，适用于处理多语种、非标准文本和具有复杂结构的语言。SentencePiece可以用于分词、标记化和构建词汇表等自然语言处理任务。
https://github.com/google/sentencepiece









# Reference

[Sequence Tagging with Tensorflow](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)

[斯坦福的CS224n Deep Learning for Natural Language Processing](http://web.stanford.edu/class/cs224n/)

https://www.tensorflow.org/tutorials/word2vec

https://github.com/graykode/nlp-tutorial

[Notes on Deep Learning for NLP](https://arxiv.org/pdf/1808.09772.pdf)

[NLP深度学习发展历史](https://ruder.io/a-review-of-the-recent-history-of-nlp/)

[Transformer、CNN、RNN的对比](https://zhuanlan.zhihu.com/p/54743941)