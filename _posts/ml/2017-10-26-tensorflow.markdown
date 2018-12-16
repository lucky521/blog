---
title: "TensorFlow实用手册"
categories: [MachineLearning]
layout: post
---

本篇所涉及的TensorFlow API都在官方文档有所涉及，https://www.tensorflow.org/api_docs/

# 框架体系

## TensorFlow Core (Low-Level API)

TensorFlow Core 指的是 low-level TensorFlow APIs。 https://www.tensorflow.org/guide/low_level_intro

Running the computational graph in a session

			tf.Graph
			tf.Session
			tf.placeholder

在TensorFlow中，数据以 tensor 为单元。tensor本质上是n维数组。
数组的维度叫做tensor的rank。一个标量是rank为0的tensor。
每个维度的数组长度组成的tuple元组叫做tensor的shape。




## TensorFlow Estimator (High-Level API)

### tf.estimator.Estimator 类

定义模型

### tf.estimator.train_and_evaluate 函数

### Spec

 tf.estimator.EstimatorSpec 用来定义Estimator的操作。

 tf.estimator.TrainSpec  用来定义输入的训练数据

 tf.estimator.EvalSpec  用来定义eval部分的配置。

### Config

 tf.estimator.RunConfig

 tf.estimator.ModeKeys  设定当前的工作模式（eval、predict、train）

 tf.estimator.WarmStartSettings







## 可视化模块 Tensorboard

https://github.com/tensorflow/tensorboard/blob/master/README.md

### tensorboard 命令

		tensorboard --logdir=/path/to/log-directory

tensorboard默认占用了6006端口

		lsof -i:6006

如果想查找当前目录里面有多少文件可以被可视化出来，可以用inspect参数来扫描目录。路径可以是相对路径。

		tensorboard --inspect  --logdir=./

### tf.summary API

tf.summary 提供了向文件写入模型内部的结构和数据信息的方法，以供 tensorboard 来展示。

https://www.tensorflow.org/api_guides/python/summary

一个例子：https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_basic.py


### 可视化中间文件 tfevent

events.out.tfevents.XXX.local 文件是summary方法所生成的文件，其中包含了用于tensorboard进行可视化展示所需的信息。每创建一个tf.summary.FileWriter实例，就会对应的生成一个tfevent文件。

event files, which contain information that TensorBoard uses to create visualizations.

Everytime when tf.summary.FileWriter is instantiated, a event file will be saved in the specified directory.


### Data 可视化

Scalar、custom_scalar、images、audio、text各种类型的数据都能通过在代码里创建summary，然后在tensorboard的相应面板里查看。

比如在代码里调用 tf.summary.scalar("loss", loss)，就能在scalars可视化面板里看到“loss”值的变化情况。

Histogram、Distribution

在代码里调用 tf.summary.histogram，就能在可视化面板里查看数据的分布。

### Model graph 可视化

https://www.tensorflow.org/guide/graph_viz

展示了整个模型的结构图。


### Precision-Recall Curve 可视化

https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/pr_curve

### Embedding Projector 可视化

Embedding Projector是Tensorboard的一个功能，可以可视化的查看embeddings。
把checkpoint文件、model.ckpt文件、metadata.tsv文件、projector_config.pbtxt文件都放在同一个目录下。
到这个目录下然后运行 tensorbord --logdir=.

metadata.tsv按顺序存储了每一个embedding的label，可以是id也是可以name。

visualize的方式有T-SNE、PCA以及custom的方式。

### Beholder 可视化

https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/beholder

### Debugger 可视化

https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/debugger

### Profile 可视化

用于监控TPU上的性能指标。

https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/profile





## 调试模块 TensorFlow Debugger

https://www.tensorflow.org/api_guides/python/tfdbg








# 重要的元素

## tf.constant 图常数

https://www.tensorflow.org/api_guides/python/constant_op

## 图变量

https://www.tensorflow.org/api_guides/python/state_ops

### tf.Variable 参数

Variable 代表着模型中的参数，算法的核心目的是在训练参数。

### tf.get_variable

tf.Variable与tf.get_variable()的区别是：
tf.get_variable() 会检查当前命名空间下是否存在同样name的变量，可以方便共享变量。而tf.Variable 每次都会新建一个变量。
使用tf.Variable时，如果检测到命名冲突，系统会自己处理。使用tf.get_variable()时，系统不会处理冲突，而会报错。

tf.contrib.layers.xavier_initializer
Returns an initializer performing "Xavier" initialization for weights.

推荐使用的初始化方法为

```
W = tf.get_variable("W", shape=[784, 256],
       initializer=tf.contrib.layers.xavier_initializer())
```

### tf.global_variables_initializer

An Op that initializes global variables in the graph.

### tf.variance_scaling_initializer


### tf.tables_initializer

Returns an Op that initializes all tables of the default graph.

## 命名空间

命名空间和给变量命名主要是有益于在tensorboard上可视化展示。

### tf.name_scope

name_scope 作用于操作。

### tf.variable_scope

variable_scope 可以通过设置 reuse 标志以及初始化方式来影响域下的变量。


## tf.placeholder 可变数据输入

在构建模型的时候没必要先把数据写好，先用tf.placeholder把数据类型确定就行。在真正session执行的时候再用feed_dict把输入填进去就好。


## tf.Operation


## tf.Session 运行数据流

在 tf.Session 之前的过程都是定义，tf.Session().run(...)才是真正执行前面定义好的操作。

Run函数 是整个tensorflow graph的核心过程。

首先看 run函数的接口

		run(
		    fetches,
		    feed_dict=None,
		    options=None,
		    run_metadata=None
		)

run函数的功能是：执行一轮图计算，执行fetches参数中的operation和计算fetches中的tensor。This method runs one "step" of TensorFlow computation, by running the necessary graph fragment to execute every Operation and evaluate every Tensor in fetches, substituting the values in feed_dict for the corresponding input values.

所以fetches参数里可以写成一个list，里面可以是Operation(比如优化器的minimize)，也可以是Tensor，也可以是Tensor所对应的名字，

这个函数的返回值含义和输入到fetches参数的名称保持一一对应。如果是Operation的话，对应返回的是None
.The value returned by run() has the same shape as the fetches argument, where the leaves are replaced by the corresponding values returned by TensorFlow.

https://www.tensorflow.org/api_docs/python/tf/Session#run








# 常用函数

用一下API完成日常工作。 包括基础操作方法、模型保存加载方法、模型流图构建方法、模型训练方法。

## 基础操作函数 Common Function

先看一些基础的操作函数。

### 基本运算函数

```
tf.random_normal

tf.random_uniform

tf.reduce_mean

tf.reduce_max

tf.reduce_min

tf.argmax(vector, dimention)：返回的是vector中的最大值的索引号

tf.multiply() 两个矩阵中对应元素各自相乘

tf.matmul() 将矩阵a乘以矩阵b，生成a * b。

tf.equal

tf.where
tf.where(condition, x = None, y = None, name = None)，根据condition判定返回。即condition是True，选择x；condition是False，选择y。

```

### 类型转换函数

https://www.tensorflow.org/api_guides/python/array_ops

```
tf.cast

tf.expand_dims

tf.reshape

```

## 模型的保存和加载

我们经常在训练完一个模型之后希望保存训练的结果，这些结果指的是模型的参数，以便下次迭代的训练或者用作测试。

一种是传统的Saver类save保存和restore恢复方法。Tensorflow针对这一需求提供了Saver类。

tf.train.get_checkpoint_state   输入路径必须是绝对路径

```
# 保存
tf.train.Saver()
save_path = saver.save(sess, model_path)
...
# 加载
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path)) # tf.train.latest_checkpoint自动获取最后一次保存的模型
saver.restore(sess, model_path)
```

还有一种是比较新颖的SavedModelBuilder类的builder保存和loader文件里的load恢复方法。

```
# 保存
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
builder.add_meta_graph_and_variables(...)
builder.save()
...
# 加载
tf.saved_model.loader.load(sess, ["tag"], export_dir)

```

### 高阶函数

tf.map_fn



## 神经网络构建函数 Build Graph

### 激活函数

```
tf.nn.relu
tf.nn.sigmoid
tf.nn.tanh
tf.nn.dropout
tf.nn.softmax
```

### 卷积函数

```
tf.nn.convolution
tf.nn.conv2d
```

### 池化函数

```
tf.nn.avg_pool
tf.nn.max_pool
```

### 正则化函数

```
tf.contrib.layers.l1_regularizer(scale, scope=None)
返回一个用来执行L1正则化的函数,函数的签名是func(weights).

tf.contrib.layers.l2_regularizer(scale, scope=None)
返回一个执行L2正则化的函数.
```


## 模型训练函数 Set Train

### 损失函数

交叉熵损失函数 softmax_cross_entropy_with_logits

softmax_cross_entropy_with_logits 是用的最多的，此外还有mean_squared_error和sigmoid_cross_entropy。

```
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_nn, labels=Y)

tf.reduce_mean(cross_entropy)
```


```
tf.distributions.kl_divergence
```

### 优化器函数

优化器函数是怎么更新整个网络参数的？
通过operation。

```
tf.train.AdamOptimizer

tf.train.GradientDescentOptimizer
```






# Tensorflow 机器学习模型

https://github.com/aymericdamien/TensorFlow-Examples/tree/master/examples/2_BasicModels

	SVM
	kmeans
	线性回归
	逻辑回归
	KNN
	随机森林






# 训练 Embeddings

Embedding是一个行为，把离线形式的事物影响到为实数向量。Embedding这个词同时也是该行为所输出的东西，我们把输出的实数向量也称作是Embedding。

An embedding is a mapping from discrete objects, such as words, to vectors of real numbers.

An embedding is a relatively low-dimensional space into which you can translate high-dimensional vectors. Embeddings make it easier to do machine learning on large inputs like sparse vectors representing words. Ideally, an embedding captures some of the semantics of the input by placing semantically similar inputs close together in the embedding space. An embedding can be learned and reused across models.

这个链接讲了我们如何用TensorFlow做embedding https://www.tensorflow.org/guide/embedding，https://github.com/tensorflow/models/tree/master/tutorials/embedding

比如我们要做word embeddings.

## Do embedding

怎么把 raw format 的 feature data 转变为 embedding format(vector<float>)的  data？

tf.nn.embedding_lookup


## Visualize your embeddings

把一个embedding在 tensorboard 上可视化出来，需要做三件事。

1) Setup a 2D tensor that holds your embedding(s).

2) Periodically save your model variables in a checkpoint in LOG_DIR.

3) (Optional) Associate metadata with your embedding.

参考https://stackoverflow.com/questions/40849116/how-to-use-tensorboard-embedding-projector





# Tensorflow 模型格式

下面两种模型文件格式对应着tensorflow的两种模型保存方式。

## checkpoint文件 和 serving pb/variable文件之间的转换

checkpoint文件 是用于本地加载模型然后进行本地预测的。
serving variable文件是用来让tensorflow serving加载并进行远程预测的。

## checkpoint文件

这是由 tf.train.Saver 类生成的模型文件。

checkpoints, which are versions of the model created during training. 存储的为最近的几次迭代保存的模型名称以及路径：

		meta file: 在meta文件中保存的为模型的图。describes the saved graph structure, includes GraphDef, SaverDef, and so on; then apply tf.train.import_meta_graph('/tmp/model.ckpt.meta'), will restore Saver and Graph.
	
		index file: 在index文件中保存的为模型参数的名称以及具体属性。it is a string-string immutable table(tensorflow::table::Table). Each key is a name of a tensor and its value is a serialized BundleEntryProto. Each BundleEntryProto describes the metadata of a tensor: which of the "data" files contains the content of a tensor, the offset into that file, checksum, some auxiliary data, etc.
	
		data file: 在data文件中保存的为模型参数的数值。it is TensorBundle collection, save the values of all variables.


https://www.tensorflow.org/guide/checkpoints


## serving pb/variable文件

这是由 tf.saved_model.builder.SavedModelBuilder 类生成的模型文件。

```
	|-- mnist_saved_model
	|   `-- 1531711208
	|       |-- saved_model.pb   保存了serialized tensorflow::SavedModel
	|       `-- variables   保存了variables
	|           |-- variables.data-00000-of-00001
	|           `-- variables.index
```


### 构建模型的输入输出以及调用方式

三种调用方式：
```
分类问题对应客户端中的classify方法
       CLASSIFY_METHOD_NAME
回归问题对于客户端中的regress方法
       REGRESS_METHOD_NAME
预测问题对应客户端中的predict方法（基本与分类问题相同，分类问题可以多一个参数“classes”）
       PREDICT_METHOD_NAME
```



下面是构建serving pb/variable文件的过程：
```
tf.saved_model.builder.SavedModelBuilder

tf.saved_model.utils.build_tensor_info

tf.saved_model.signature_def_utils.build_signature_def

builder.add_meta_graph_and_variables

builder.save()
```








# Tensorflow 样本数据格式 - TFRecord

TFRecord是Tensorflow特有的二进制数据存储格式。它的好处是性能，在加载和传输时代价较小。另一个好处是可以存储序列化数据。

我们用Tensorflow API可以方便的构建和读写TFRecord数据。


tf.train.Example

TFRecord是文件形态，tf.train.Example就是内存对象形态


tf.train.Feature


tf.python_io.TFRecordWriter









# TensorFlow Serving

TensorFlow Serving 是基于 gRPC 和 Protocol Buffers 开发的。
https://github.com/tensorflow/serving ,
https://www.tensorflow.org/serving/serving_basic

## 服务端 tensorflow-model-server

在服务器端安装好之后，核心就是 tensorflow_model_server 这个binary。

		tensorflow_model_server --help

1. 在服务端先要训练一个模型

可以用 models repo 中的例子：

		cd models/official/mnist
		python mnist.py --export_dir ./tmp/mnist_saved_model

或者用 tensorflow_serving repo中的例子：

		cd tensorflow_serving/example/
		python mnist_saved_model.py ./tmp/mnist_model

2. 保存的模型是这样子的：

```
	|-- mnist_saved_model
	|   `-- 1531711208
	|       |-- saved_model.pb   保存了serialized tensorflow::SavedModel
	|       `-- variables   保存了variables
	|           |-- variables.data-00000-of-00001
	|           `-- variables.index
```

3. 然后将这个模型载入到 TensorFlow ModelServer，注意输入的模型路径必须是绝对路径。

    tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/


## 客户端 tensorflow-serving-api

在客户端把样本数据作为请求发送到TensorFlow ModelServer，

		python tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=localhost:9000





## TensorFlow Serving 客户端-服务端数据交互格式

TensorProto
TensorInfo是一个pb message，定义在tensorflow/core/framework/tensor.proto，用来表示一个Tensor。


SignatureDef
由inputs TensorInfo、outputs TensorInfo、method_name三个成员构成。

SignatureDefMap
由name->SignatureDef 构成的map。

MetaGraphDef
表示一个模型的graph。


PredictRequest
由 map<string, TensorProto> 作为请求输入。

PredictResponse
由 map<string, TensorProto> 作为请求返回。















# 分布式TensorFlow集群 - Distributed TensorFlow

	TensorFlow server - tf.train.Server instance
	
		Master service
	
		Worker service
	
	Client - 在单例环境中一个graph位于一个tensorflow::Session中。对于分布式环境中，Session位于一个Server中。
	
	Cluster - tf.train.ClusterSpec object 用于在创建 tf.train.Server 时指明spec。
	
	Job - 一个Cluster可能包含多个Job。
	
	Task - 一个Job可能有多个Task。


tf.train.Server.create_local_server 单进程集群，这主要是其演示作用吧。

tf.train.ClusterSpec  创建cluster配置描述

tf.train.Server 创建server实例

在模型中指明在特定节点或设备进行某个操作

		with tf.device("/job:ps/task:1"):
		  weights_2 = tf.Variable(...)
		  biases_2 = tf.Variable(...)










# Tensorflow 训练示例

本节贴出了一些Tensorflow在常见训练集数据下的训练过程。

## MNIST 手写字数据集模型训练

MNIST数据集是一个手写阿拉伯数字0-9的图像素材库，它包含60000个训练样本集和10000个测试样本集。我们可以去官网下载素材库，也可以直接使用TensorFlow以package引用形式提供的MNIST。

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist


	fully_connected_feed.py 采用Feed数据方式训练。
	mnist_softmax.py  使用Softmax回归训练。
	mnist_softmax_xla.py 采用XLA框架训练。
	mnist_with_summaries.py



https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks

	autoencoder.py 使用Autoencoder训练。
	neural_network_raw.py 	2-Hidden Layers Fully Connected Neural Network
	recurrent_network.py 使用RNN训练。
	convolutional_network.py  using TensorFlow layers API
	convolutional_network_raw.py
	gan.py


https://github.com/nlintz/TensorFlow-Tutorials/blob/master/05_convolutional_net.py

	05_convolutional_net.py 使用CNN训练。



## Fashion-MNIST 数据集

这是一个服饰类的图像数据集，包含了10个类别，分别是10种服饰类型。




## ImageNet 图像数据集模型训练

ImageNet的图像分类数据集叫做ILSVRC。ILSVRC图像分类数据集包含了来自1000个类别的120万张图片，其中每张图片属于且只属于一个类别。

### GoogLeNet

https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py

### AlexNet

https://github.com/tensorflow/models/blob/master/tutorials/image/alexnet/alexnet_benchmark.py



### LFW 人脸数据集模型训练

LFW 是人脸识别数据集，英文全称是Labeled Faces in the Wild，所有人脸样本都加了标签。

https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

### FaceNet

FaceNet启发于OpenFace项目，使用TensorFlow创建的人脸识别框架。

https://github.com/davidsandberg/facenet



## 语音数据集训练模型

### spoken numbers pcm 数据集

https://github.com/pannous/tensorflow-speech-recognition

### WaveNet

https://deepmind.com/blog/wavenet-generative-model-raw-audio/
