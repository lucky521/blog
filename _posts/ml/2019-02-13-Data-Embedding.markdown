---
layout: post
title:  "Data Embedding"
subtitle: ""
categories: [MachineLearning]
---

# Data embedding是什么？
Embedding的概念来自于word embeddings。
Embedding is a transformation from discrete values/scalars to dense real value vectors. 有的地方把embedding翻译为嵌套，有的地方把它翻译为向量。


Embedding是一个行为，把离线形式的事物影响为实数向量。Embedding这个词同时也是该行为所输出的东西，我们把输出的实数向量也称作是Embedding。

An embedding is a mapping from discrete objects, such as words, to vectors of real numbers.

An embedding is a relatively low-dimensional space into which you can translate high-dimensional vectors. Embeddings make it easier to do machine learning on large inputs like sparse vectors representing words. Ideally, an embedding captures some of the semantics of the input by placing semantically similar inputs close together in the embedding space. 
An embedding can be learned and reused across models.


它可以单独使用来学习一个单词嵌入，以后可以保存并在另一个模型中使用，比如作为特征。
它可以用作深度学习模型的一部分，其中嵌入与模型本身一起学习。
它可以用来加载预先训练的词嵌入模型，这是一种迁移学习。
它可以用来做相似召回，在某种空间计算embedding的相似度。


广告、推荐、搜索等领域用户数据的稀疏性几乎必然要求在构建DNN之前对user和item进行embedding后才能进行有效的训练。


# 神经网络中的Embedding layer

在Keras中有专门的Embedding层，其作用是：Turns positive integers (indexes) into dense vectors of fixed size. 
eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
This layer can only be used as the first layer in a model.
```
keras.layers.Embedding(input_dim,
        output_dim,
        embeddings_initializer='uniform',
        embeddings_regularizer=None,
        activity_regularizer=None, 
        embeddings_constraint=None, 
        mask_zero=False, 
        input_length=None)

```

对应的TFlearn中的Embedding层:
```
tflearn.layers.embedding_ops.embedding(incoming,
        input_dim, 
        output_dim, 
        validate_indices=False, 
        weights_init='truncated_normal', 
        trainable=True, 
        restore=True, 
        reuse=False, 
        scope=None, 
        name='Embedding')
```

在Tensorflow中也可以在网络结构中加入Embedding层：
```
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs) # lookup table
```


## Embedding层的输入：

嵌入层被定义为网络的第一个隐藏层。它必须指定3个参数：

    - input_dim：这是文本数据中词汇的取值可能数。例如，如果您的数据是整数编码为0-9之间的值，那么词汇的大小就是10个单词；
    - output_dim：这是嵌入单词的向量空间的大小。它为每个单词定义了这个层的输出向量的大小。例如，它可能是32或100甚至更大，可以视为具体问题的超参数；
    - input_length：这是输入序列的长度，就像您为Keras模型的任何输入层所定义的一样，也就是一次输入带有的词汇个数。例如，如果您的所有输入文档都由1000个字组成，那么input_length就是1000。

被Embedding的对象（比如word）必须是有限个数的。embedding层要求输入数据是整数编码的，所以每个word都用一个唯一的整数表示。这个数据准备步骤可以使用Keras提供的Tokenizer API来执行。


## Embedding层的输出：

嵌入层的输出是一个二维向量，每个word在输入文本（输入文档）序列中嵌入一个。





# 训练 Embeddings

怎么把 raw format 的 feature data 转变为 embedding format(也就是浮点数向量vector<float>) 的 embedding data？

下面链接讲了我们如何用TensorFlow做embedding 
https://www.tensorflow.org/guide/embedding  
下面两个链接讲的典型的word embedding，即word2vec。
https://github.com/tensorflow/models/tree/master/tutorials/embedding  
https://www.tensorflow.org/tutorials/representation/word2vec


嵌入层用随机权重进行初始化，并将学习训练数据集中所有单词的嵌入。



首先要有语料库，把它切分为word，每个word赋予一个int作为id。
比如语料“I have a cat.”， [“I”, “have”, “a”, “cat”, “.”]
vocabulary_size = 5
embedding_size = len(embedding-vector)
word_ids = [1,2,3,4,5]
embedded_word_ids = [[1, xxx], [2, yyy]...,[5, zzz]]

```
word_embeddings = tf.get_variable(“word_embeddings”, [vocabulary_size, embedding_size])
embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)
```


tf.nn.embedding_lookup 这个函数到底做了什么？https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do









# Visualize your embeddings

把一个embedding在 tensorboard 上可视化出来，需要做三件事。

1) Setup a 2D tensor that holds your embedding(s).

2) Periodically save your model variables in a checkpoint in LOG_DIR.

3) (Optional) Associate metadata with your embedding.

参考https://stackoverflow.com/questions/40849116/how-to-use-tensorboard-embedding-projector









# Embedding在推荐和排序中的应用

word2vec(query2vec)

item2vec(doc2vec)

user2vec


## 怎么训练和生成query的Embedding？



## 怎么训练和生成item的Embedding？




## 怎么训练和生成user的Embedding？

item和user的量可以认为是无限的，所以不能直接使用它们的index来构建。我们可以用其某些有限的属性来表达它们。比如，说user，其实聊的是user喜欢什么item，接触过什么item，那么其中的核心其实还是item，这样理解的话，user的Embedding其实就源自于若干个与之相关的item的Embedding。



## 怎么把word、item、user的Embedding训练到同一个维度？







# Embedding 论文

Embedding是一种方法，而它不是直接去解决目标问题的模型，但有了它作为模型或者输入的一部分，需要问题能够方便的求解。

Word-embedding是基础: https://paperswithcode.com/task/word-embeddings

Item-embedding:

Query-embedding:

User-embedding:


## Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba

## Item2Vec - Neural Item Embedding for Collaborative Filtering

## Real-time Personalization using Embeddings for Search Ranking at Airbnb

## Learning Item-Interaction Embeddings for User Recommendations




## Youtube - Deep Neural Networks for YouTube Recommendations

user embedding就是网络的最后一个隐层，video embedding是softmax的权重

将最后softmax层的输出矩阵的列向量当作item embedding vector，而将softmax之前一层的值当作user embedding vector。

## Youtube - Latent Cross Making Use of Context in Recurrent Recommender Systems








# 相关资料

https://gist.github.com/nzw0301/333afc00bd508501268fa7bf40cafe4e



