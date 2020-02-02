---
layout: post
title:  "Data Embedding"
subtitle: "用向量表达一切"
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

## 使用Embedding的好处

最大的好处是有利于捕捉特征值之间的相似关系。
第二个好处是将sparse特征转变为dense特征，减少模型计算量。


# 神经网络中的Embedding layer

深度学习中设计离散特征的话一般都处理成Embedding的形式，作为网络的底部（第一层），一般对整体网络效果有着重要的作用。

## Embedding layer in Keras
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
## Embedding layer in TFlearn
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
## Embedding layer in Tensorflow
在Tensorflow中也可以在网络结构中加入Embedding层：
```
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
...
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



# 从哪里学习Embedding？

在word2vec中，学习的目标是一个word的Embedding表达，文本语料是学习的来源，我们通过一个word的context来学习这个word的表达，context指的是一段语料中某word相邻的words。
在广告推荐等领域中，如果要做item Embedding，那么context可以是一个用户点击行为中某被点击item相邻的被点击items。



# 如何让Embedding向量学到东西 （训练 Embeddings）

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

embeddings is a matrix where each row represents a word vector.

embedding_lookup is a quick way to get vectors corresponding to train_inputs.


tf.nn.embedding_lookup 这个函数到底做了什么？https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do
embedding_lookup不是简单的查表，id对应的向量是可以训练的，训练参数个数应该是 category num * embedding size，也就是说lookup是一种全连接层。


## 构建怎么样的网络结构，才能让Embedding layer学到输入数据的 Representation？


下面是word2vec实现的最简单的版本，这里只展示网络结构的部分。
```
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
# need to shape [batch_size, 1] for nn.nce_loss
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# Ops and variables pinned to the CPU because of missing GPU implementation
with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs) # lookup table

# Construct the variables for the NCE loss
nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

# Compute the average NCE loss for the batch.
# This does the magic:
#   tf.nn.nce_loss(weights, biases, inputs, labels, num_sampled, num_classes ...)
# It automatically draws negative samples when we evaluate the loss.
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed, num_sampled, voc_size))
# Use the adam optimizer
train_op = tf.train.AdamOptimizer(1e-1).minimize(loss)
```



# 语料中数据量较少的word，能否学到合适的Embedding值？

方法一：对稀疏的id做聚类处理




# 评估Embedding向量的效果 （Visualize your embeddings）

把一个embedding在 tensorboard 上可视化出来，需要做三件事。

1) Setup a 2D tensor that holds your embedding(s).

2) Periodically save your model variables in a checkpoint in LOG_DIR.

3) (Optional) Associate metadata with your embedding.

参考https://stackoverflow.com/questions/40849116/how-to-use-tensorboard-embedding-projector

## t-SNE dimensionality reduction technique

## Analogical Reasoning






# Embedding在推荐、排序、广告中的应用

word2vec(query2vec)

item2vec(doc2vec)

user2vec


## 怎么训练和生成item的Embedding？



## 怎么训练和生成query的Embedding？

query中每一个词蕴含的信息可以通过训练数据中由其召回的item所表达。

query的Embedding是由其切词后各个分词的Embedding所结合而成的。


## 怎么训练和生成user的Embedding？

item和user的量可以认为是无限的，所以不能直接使用它们的index来构建。我们可以用其某些有限的属性来表达它们。比如，说user，其实聊的是user喜欢什么item，接触过什么item，那么其中的核心其实还是item，这样理解的话，user的Embedding其实就源自于若干个与之相关的item的Embedding。

这里的核心就在于，如何结合用户有过行为的若干个item的Embedding，合成一个User Embedding。



## 多实体embedding向量空间一致性问题： 怎么把query、item、user的Embedding训练到同一个维度？

将word embedding和item embedding放到同一个网络里训练。也就意味着使用同一个语料进行训练。










# Embedding 相关论文

Embedding是一种方法，而它不是直接去解决目标问题的模型，但有了它作为模型或者输入的一部分，需要问题能够方便的求解。

Word-embedding是基础: https://paperswithcode.com/task/word-embeddings

Item-embedding:

Query-embedding:

User-embedding:


## Alibaba - Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba

## Alibaba - Learning and Transferring IDs Representation in E-commerce

解读：https://zhuanlan.zhihu.com/p/56119617


## Youtube - Deep Neural Networks for YouTube Recommendations

user embedding就是网络的最后一个隐层，video embedding是softmax的权重.

将最后softmax层的输出矩阵的列向量当作item embedding vector，而将softmax之前一层的值当作user embedding vector。

## Youtube - Latent Cross Making Use of Context in Recurrent Recommender Systems


## Item2Vec - Neural Item Embedding for Collaborative Filtering

主要做法是把item视为word，用户的行为序列视为一个集合，item间的共现为正样本，并按照item的频率分布进行负样本采样，缺点是相似度的计算还只是利用到了item共现信息，1).忽略了user行为序列信息; 2).没有建模用户对不同item的喜欢程度高低。

Item2vec中把用户浏览的商品集合等价于word2vec中的word的序列.

## Real-time Personalization using Embeddings for Search Ranking at Airbnb

## Learning Item-Interaction Embeddings for User Recommendations





# 构建Embedding的开源框架

## Facebook - starspace

https://github.com/facebookresearch/starspace

这个命令行工具用起来很简单：input.txt中每一行是一个session中的item序列。

./starspace train -trainFile input.txt -model pagespace -label 'page' -trainMode 1


## Flair
https://github.com/zalandoresearch/flair
A text embedding library



## fasttext


## PyTorch-BigGraph (PBG)
https://github.com/facebookresearch/PyTorch-BigGraph




# Reference

https://gist.github.com/nzw0301/333afc00bd508501268fa7bf40cafe4e



