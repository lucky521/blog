---
title: "TensorFlow实用手册"
categories: [MachineLearning]
layout: post
---

本篇所涉及的TensorFlow API都在官方文档有所涉及，https://www.tensorflow.org/api_docs/

首先要注意，tensorflow版本之间差异比较大，一些API会发生增减或者位置迁移。
```python
python -c 'import tensorflow as tf; print(tf.__version__)'
python -c 'import tensorflow as tf; print(tf.test.is_gpu_available())'
python -c 'import tensorflow as tf; tf.config.list_physical_devices()'
```


# 功能体系

TF存在几套不同方式的使用方式API。

* tf.contrib，是TF官方团队之外的贡献者贡献的功能。
* tf.compat 是为了兼容TF1和TF2的一些api

## TensorFlow Core (Low-Level API)

TensorFlow Core 指的是 low-level TensorFlow APIs。 
https://www.tensorflow.org/guide/low_level_intro

Running the computational graph in a session

			tf.Graph
			tf.Session
			tf.placeholder

在TensorFlow中，数据以 tensor 为单元。tensor本质上是n维数组。
数组的维度叫做tensor的rank。一个标量是rank为0的tensor。
每个维度的数组长度组成的tuple元组叫做tensor的shape。


## TensorFlow Estimator (High-Level API)

最适合用于模型实践的API就是tf.estimator这一套方法。 在高阶API中不显式的出现session.run的概念。

### tf.estimator.Estimator 类

tf.estimator是一个基类。

可以使用原生预设的模型子类，比如 DNNClassifier、 DNNRegressor等。也可以基于基类自己实现子类。

```python
predictor = tf.estimator.Estimator(
        model_fn=model.model_fn,
        params={  # dict of hyper parameters that will be passed into model_fn
            'feature_columns': columns,
            'config': config,
            'args': args,
        },
        model_dir=config.model_path,  如果该目录下有之前的checkpoints，会自动继续上一次的模型开始 训练。
        log_dir =config.tensorboard_dir,
        config=run_config,  # tf.estimator.RunConfig
        warm_start_from=ws  # tf.estimator.WarmStartSettings
      )
```
其中的 params["feature_columns"]是在模型中的所有FeatureColumns组成的列表或字典。后面会有一章单独讲Feature Columns。

### model_fn 和 EstimatorSpec

我们需要实现 model_fn 方法来创建EstimatorSpec并返回。模型网络的结构体现在该函数中。
在训练时，以下参数将传递给该函数。

    features: 这是来自于input_fn的特征数据。This is the first item returned from the input_fn passed to train, evaluate, and predict. This should be a single tf.Tensor or dict of same.

    labels: 这是来自于input_fn的labels数据。This is the second item returned from the input_fn passed to train, evaluate, and predict. This should be a single tf.Tensor or dict of same (for multi-head models). If mode is tf.estimator.ModeKeys.PREDICT, labels=None will be passed. If the model_fn's signature does not accept mode, the model_fn must still be able to handle labels=None.

    mode: Optional. Specifies if this training, evaluation or prediction. See tf.estimator.ModeKeys.

    params: 这个参数是我们在构建Estimator的时候传给params的内容。Optional dict of hyperparameters. Will receive what is passed to Estimator in params parameter. This allows to configure Estimators from hyper parameter tuning.

    config: Optional estimator.RunConfig object. Will receive what is passed to Estimator as its config parameter, or a default value. Allows setting up things in your model_fn based on configuration such as num_ps_replicas, or model_dir.

    Returns: tf.estimator.EstimatorSpec

tf.estimator.EstimatorSpec 用来定义Estimator的操作。该对象会作为 model_fn 参数来构建 Estimator.
它定义了一个具体的模型对象。

1. 如果是train任务，需要输入loss和train_op来构建 EstimatorSpec
train_op指的就是优化器进行最优化求解所对应的op。
2. 如果是eval任务，需要输入loss和eval_metric_ops 来构建 EstimatorSpec
eval_metric_ops 由若干tf.metrics指标模块所组成的字典，比如tf.metrics.accuracy, tf.metrics.precision, tf.metrics.recall, tf.metrics.auc.
3. 如果是predict任务，需要输入predictions 来构建 EstimatorSpec


### tf.estimator.train_and_evaluate 函数

这个方法是真正去训练模型。它的输入是 ` Estimator对象 + TrainSpec对象 + EvalSpec对象 `。

```python
# 上一节代码里创建有 estimator
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=train_hooks, max_steps=1000)

eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, hooks=eval_hooks, steps=eval_config.eval_steps)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

### input_fn Spec 

我们需要实现 input_fn 方法来创建TrainSpec和EvalSpec并返回，用于tf.estimator.train_and_evaluate.

tf.estimator.TrainSpec  用来定义输入的训练数据，需要传入 input_fn=train_input_fn.  
  * max_steps如果是None，训练就会持续进行，如果是x，就会拿训练数据训练x次(分布式训练时每个worker都训练x次)。 
  * train_hooks.append(tf.train.StopAtStepHook(num_steps=config.stop_at_steps))的作用是

tf.estimator.EvalSpec  用来定义eval部分的配置，需要传入 input_fn=eval_input_fn.
  * hooks=eval_hooks, 它是eval_hooks.append(添加一个tf.train.SessionRunHook的子类)
  * steps=eval_config.eval_steps 调用evaluate的step周期

input_fn 作为TrainSpec/EvalSpec最重要的输入参数，它是一个方法，该方法最终应该返回是数据。可以支持的类型有两种：

 1. A tuple (features, labels):

 2. A 'tf.data.Dataset' object: 该Dataset的返回值要是 tuple (features, labels) 


### Config

tf.estimator.RunConfig 各种配置都填在这个类对象中。它会被作为conf参数用于构建 Estimator。

tf.estimator.ModeKeys  设定当前的工作模式（eval、predict、train）

tf.estimator.WarmStartSettings 它被作为warm_start_from参数用于构建 Estimator。

tf.estimator.VocabInfo  表示 WarmStartSettings 的词汇信息。它被用于构建WarmStartSettings.



## Keras API (High-Level API)

```python
# 定义模型
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# 定义学习过程
model.compile(optimizaer='sgd', loss='mean_squared_error')
# 训练
xs = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float) 
ys = np.array([-3.0. -1.0, 1.0, 3.0], dtype=float)
model.fit(xs, ys, epochs=500)
# 预测
result = model.predict([4.0])
```


## Eager 模式 API

无需构建图：操作会返回具体的值，而不是构建以后再运行的计算图.

https://www.tensorflow.org/guide/eager






## 可视化模块 Tensorboard

官方文档：https://github.com/tensorflow/tensorboard/blob/master/README.md

### tensorboard 启动命令

		tensorboard --logdir=/path/to/log-directory --port=8008

tensorboard默认占用了6006端口

		lsof -i:6006

如果想查找当前目录里面有多少文件可以被可视化出来，可以用inspect参数来扫描目录。路径可以是相对路径。

		tensorboard --inspect  --logdir=./

### tf.summary 信息输出API

tf.summary 提供了向文件写入模型内部的结构和数据信息的方法，以供 tensorboard 来展示。

tf.summary.merge_all()  可以将所有summary全部保存到磁盘，以便tensorboard显示.

tf.summary.FileWriter('xxx', sess.graph)

https://www.tensorflow.org/api_guides/python/summary

一个例子：https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_basic.py


### 可视化中间文件 tfevent

events.out.tfevents.XXX.local 文件是summary方法所生成的文件，其中包含了用于tensorboard进行可视化展示所需的信息。
每创建一个tf.summary.FileWriter实例，就会对应的生成一个tfevent文件。


### Data 可视化

Scalar、custom_scalar、images、audio、text各种类型的数据都能通过在代码里创建summary，然后在tensorboard的相应面板里查看。

- Scalar 标量可视化
  比如在代码里调用 tf.summary.scalar("loss", loss)，就能在scalars可视化面板里看到“loss”值的变化情况。

- Histogram、Distribution 分布可视化
  “Distributions” tab contains a plot that shows the distribution of the values of the tensor (y-axis) through steps (x-axis).

  在代码里调用 tf.summary.histogram，就能在可视化面板里查看数据的分布。

  HISTOGRAMS 和 DISTRIBUTIONS 是对同一数据不同方式的展现。与 DISTRIBUTIONS 不同的是，HISTOGRAMS 可以说是 频数分布直方图 的堆叠。


### Model graph 可视化

https://www.tensorflow.org/guide/graph_viz

展示了整个模型的结构图。

### Precision-Recall Curve 可视化

https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/pr_curve

### Projector 可视化

Embedding Projector是Tensorboard的一个功能，可以可视化的查看embeddings。
- 把checkpoint文件、model.ckpt文件、metadata.tsv文件、projector_config.pbtxt文件都放在同一个目录下。到这个目录下然后运行 tensorbord --logdir=.

  metadata.tsv按顺序存储了每一个embedding的label，可以是id也是可以name。

visualize的方式有T-SNE、PCA以及custom的方式。

kernel
kernel/Adam

### Beholder 可视化

https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/beholder

### Debugger 可视化

https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/debugger

### Profile 可视化

用于监控TPU上的性能指标。

https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/profile





## 调试模块 TensorFlow Debugger

https://www.tensorflow.org/api_guides/python/tfdbg



# 重要元素

## Tensor、 Graph、 Session 是Tensorflow体系三个非常重要的概念

tensorflow::Tensor 是模型输入、模型输出、模型内部任何流动数据的载体。
tensorflow::GraphDef是图、模型的载体。
tensorflow::Session是训练或预测过程的会话载体。

```
  // Construct your graph.
  tensorflow::GraphDef graph = ...;

  // Create a Session running TensorFlow locally in process.
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession({}));

  // Initialize the session with the graph.
  tensorflow::Status s = session->Create(graph);
  if (!s.ok()) { ... }

  // Specify the 'feeds' of your network if needed.
  std::vector<std::pair<string, tensorflow::Tensor>> inputs;

  // Run the session, asking for the first output of "my_output".
  std::vector<tensorflow::Tensor> outputs;
  s = session->Run(inputs, {"my_output:0"}, {}, &outputs);
  if (!s.ok()) { ... }

  // Do something with your outputs
  auto output_vector = outputs[0].vec<float>();
  if (output_vector(0) > 0.5) { ... }

  // Close the session.
  session->Close();
```

## Tensor

1. Tensor 是Tensorflow中承载多维数据的容器。

这里的tensor在形式上就是 dense tensor.

把原始数据转变为tensor
```python
def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
  input_name = "file_reader"
  file_reader = tf.read_file(file_name, input_name)
  image_reader = tf.image.decode_jpeg(file_reader, channels = 3, name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)
```

2. SparseTensor 是稀疏形式表示的向量容器。

它由三个tensor组成，  `indices`, `values`, and `dense_shape`.

定义在tensorflow/python/framework/sparse_tensor.py


## Graph 图

三种形态的“图”：

1. - tf.Graph： 运行状态的Graph， 被定义为“一些 Operation 和 Tensor 的集合”。
真正加载到内存处于可运行状态(训练、预测)时的graph。

2. - tf.GraphDef： 序列化状态的GraphDef，它可以被存储到pb文件中，然后在需要时从pb文件加载。
The GraphDef format is a version of the ProtoBuf serialization protocol, in either text or binary, that encodes the definition of a TensorFlow graph.
A GraphDef can also include the weights of a trained model as we will see later, but it doesn’t have to — the weights can be stored as separate checkpoint files.

3. - tf.MetaGraphDef: PB形式表示的GraphDef,其中包含了图结构、权值、assets 和 SignatureDef。
MetaGraph is a dataflow graph, plus its associated variables, assets, and signatures. A MetaGraphDef is the protocol buffer representation of a MetaGraph.
定义在 tensorflow/core/protobuf/meta_graph.proto


三种“图”对应的API：

1. - tf.train.Saver() / saver.restore()
tf.train.saver.save() 在保存check-point的同时也会保存Meta Graph。但是在恢复图时，tf.train.saver.restore() 只恢复 Variable.
如果要从MetaGraph恢复图，需要使用 import_meta_graph。

Meta Graph中虽然包含Variable的信息，却没有 Variable 的实际值。所以从Meta Graph中恢复的图，其训练是从随机初始化的值开始的。训练中Variable的实际值都保存在check-point中，如果要从之前训练的状态继续恢复训练，就要从check-point中restore。

2. - tf.train.write_graph() / tf.import_graph_def()

下面是从pb文件加载tf.Graph的例子

```
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()
  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
  return graph
```

3. - tf.train.export_meta_graph() / tf.train.import_meta_graph()

参考 tensorflow/python/framework/meta_graph.py


## 图常数

tf.constant
constant()是个常量构造函数，它可以用在assign/initilizer中为Variable生成数据,也可以用在feed_dict{}中为placeholder生成数据。

https://www.tensorflow.org/api_guides/python/constant_op

## 图变量

https://www.tensorflow.org/api_guides/python/state_ops

### tf.Variable 参数

Variable 代表着模型中的参数，算法的核心目的是在训练参数，也就是不断的修正所有的tf.Variable。
tf.Variable的type和shape是在初始化时确定的，之后不可以再修改；value一般在初始化时随机给出，在训练时不断更新。

### tf.get_variable

tf.Variable与tf.get_variable()的区别是：
tf.get_variable() 会检查当前命名空间下是否存在同样name的变量，可以方便共享变量。而 tf.Variable 每次都会新建一个变量。
使用tf.Variable时，如果检测到命名冲突，系统会自己处理。使用tf.get_variable()时，系统不会处理冲突，而会报错。

tf.contrib.layers.xavier_initializer
Returns an initializer performing "Xavier" initialization for weights.

推荐使用的初始化方法为

```python
W = tf.get_variable("W", shape=[784, 256],
       initializer=tf.contrib.layers.xavier_initializer())
```

### tf.global_variables_initializer

An Op that initializes global variables in the graph.

一般在图中所有tensor variables(tf.Variable)都定义好之后才初始化。这个函数返回的是初始化全局变量的OP。
一般使用session.run来初始化这个op。
```
init = tf.global_variables_initializer()
sess.run(init)
```

### tf.variance_scaling_initializer


### tf.tables_initializer

Returns an Op that initializes all tables of the default graph.


### tf.reset_default_graph



## 命名空间

命名空间和给变量命名主要是有益于在tensorboard上可视化展示。

### tf.name_scope

name_scope 作用于操作。

### tf.variable_scope

variable_scope 可以通过设置 reuse 标志以及初始化方式来影响域下的变量。

## Collection

tensorflow的collection提供一个全局的存储机制，不会受到变量名生存空间的影响。一处保存，到处可取。

## tf.placeholder 可变数据输入

在构建模型的时候没必要先把数据写好，先用tf.placeholder把数据类型确定就行。在真正session执行的时候再用feed_dict把输入填进去就好。


## tf.Operation

运算节点（也有人称作是 算子）。Operation节点的输入是tensor或0，输出是tensor或0.

[不同类型OP](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/1.png)

在graph.pbtxt文件中能看到每一个node里，都有一个key名为op的字段，它指明了对tensor对象的操作。

The `local_init_op` is an `Operation` that is run always after a new session was created.

### get_operation_by_name 从graph中由名字获取到op
graph.get_operation_by_name(op_name)


## custom_ops

custom op指的是使用C++来实现自己的tensor操作。 

当然了Tensorflow 内部的运算方法也都是通过OP这一方式来注册的。

https://www.tensorflow.org/guide/extend/op

https://github.com/tensorflow/custom-op


_as_variant_tensor


### 定义自定义op的接口
```cpp
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
using namespace tensorflow;

REGISTER_OP("接口名称")
    .Input("输入名称: 类型int32")
    .Output("输出名称: 类型int32")
    .Attr("属性名称: 属性约束")
    .SetIsStateful() // prevents constant folding
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("接口名称").Device(DEVICE_CPU), 类名);
```
- 其中input和output都是在定义输入输出tensor的名称和类型
- 其中还实现了一个Shape functions: infers the shape of outputs given that of inputs.

### 编写自定义op类的内部实现

After you define the interface, provide one or more implementations of the op. To create one of these kernels, create a class that extends OpKernel and overrides the Compute method. The Compute method provides one context argument of type OpKernelContext*, from which you can access useful things like the input and output tensors.

需要实现一个compute方法
取到输入tensor： context->input
分配出输出tensor： allocate_output


```cpp
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};
```

### 直接用g++编译自定义op

```shell
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

### 使用bazel编译自定义op

tf_custom_op_library(
    name = "想编译的.so",
    srcs = ["想编译的.cc"],
)

### tf.load_op_library 加载自定义op

使用 tf.load_op_library 加载自己编译的so.

### 模型设计中使用自定义op计算

```python
def testShuffle(self):
        shuffle_module = tf.load_op_library('shuffle_op.so')
        shuffle = shuffle_module.shuffle

        input_tensor = np.arange(12).reshape((3, 4))
        desired_shape = np.array([6, -1])
        output_tensor = input_tensor.reshape((6, 2))
        with self.test_session():
            result = shuffle(input_tensor, desired_shape)
            self.assertAllEqual(result.eval(), output_tensor)

        input_tensor = np.arange(12).reshape((3, 4))
        desired_shape = np.array([5, -1])
        output_tensor = input_tensor.reshape((6, 2))[:-1]
        with self.test_session():
            result = shuffle(input_tensor, desired_shape)
            self.assertAllEqual(result.eval(), output_tensor) 
```

更多例子：https://www.programcreek.com/python/example/90369/tensorflow.load_op_library

- 这里python方法加载了so之后，对应的方法名如何确定？

REGISTER_KERNEL_BUILDER 和 REGISTER_OP 后面跟的接口名称是"若干个首字母大写的单词"组成的名称，它对应到python之后，接口名称就变为"每个单词全变为小写，单词间以下划线分割"的名称。这应该是swig处理的。


### python加载带有custom op的saved_model

```python
op = tf.load_op_library('./xxxx.so')

tensorflow.python.saved_model.load(模型文件)
```


### Dataset ops

实现一个 from tensorflow.python.data.ops import dataset_ops 的子类，然后将该类对象传入到 input_fn .

### CustomOP是如何存入到导出模型中 

首先在 saved_model.pbtxt 文件中可以到对应的op名称。应该说这些op的代码并没有保存到模型中，而必须要让加载模型的程序提前加载好这些custom op。








## tf.Session 运行数据流

Session是一个运动状态。图的运行只发生在会话中，开启会话后，就可以用数据填充节点，进行运算；关闭会话后，就不能计算。

在 tf.Session 之前的过程都是定义，tf.Session().run(...)才是真正执行前面定义好的操作。如果只声明tensor而运行session.run，是不会运行计算逻辑的。

Run函数 是整个tensorflow graph运动的核心过程。

首先看 run函数的接口
```python
		run(
		    fetches,
		    feed_dict=None,
		    options=None,
		    run_metadata=None
		)
```

run函数的功能是：执行一轮图计算，执行fetches参数中的operation和计算fetches中的tensor。This method runs one "step" of TensorFlow computation, by running the necessary graph fragment to execute every Operation and evaluate every Tensor in fetches, substituting the values in feed_dict for the corresponding input values. 

所以fetches参数里可以写成一个list，里面可以是Operation(比如优化器的minimize)，也可以是Tensor，也可以是Tensor所对应的名字.

这个函数的返回值含义和输入到fetches参数的名称保持一一对应。
如果是Operation的话，对应返回的是None.
The value returned by run() has the same shape as the fetches argument, where the leaves are replaced by the corresponding values returned by TensorFlow. 
返回的值是由tensor组成的数组。

如果是global_variables_initializer（这也是一个op）的返回值的话，就是在图中初始化所有变量。

如果fetches是优化器的话，就会更新网络权值。

feed_dict中添加真实数据用来填充tf.placeholder，这样才能更新网络权值。

https://www.tensorflow.org/api_docs/python/tf/Session#run



## tf.app.run

tf.app.run是TensorFlow程序的入口。

```python
import tensorflow as tf
import argparse #导入命令行解析模块
import sys

FLAGS=None
def main(_):
    print(sys.argv[0])

if __name__=="__main__": #用这种方式保证了，如果此文件被其他文件import的时候，不会执行main中的代码
    #创建对象
    parse=argparse.ArgumentParser()
    #增加命令行
    parse.add_argument('--dataDir',type=str,default='\\tmp\\tensorflow\\mnist\\inputData',
                    help='Directory for string input data')
    FLAGS, unparsed=parse.parse_known_args()
```


## tf.function
Tensorflow 2.0引入了tf.function这一概念. 为的是移除tf.Session这一概念．这样可以帮助用户更好的组织代码，不用将tf.Session作为一个变量在Python函数中传来传去，我们可以用一个Python装饰符来进行加速，那就是@tf.function




# 常用函数

用以下API完成日常工作。 包括基础操作方法、模型保存加载方法、模型流图构建方法、模型训练方法。

## 基础操作函数 Common Function

先看一些基础的操作函数。

### 基本运算函数

- tensor向量随机生成
```
tf.random_normal

tf.random_uniform

tf.truncated_normal(shape, mean, stddev)
shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正态分布，均值和标准差自己设定
```

- 沿tensor向量某一个维度的计算
```
tf.reduce_mean: 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。

tf.reduce_max: 计算tensor指定轴方向上的各个元素的最大值;

tf.reduce_min: 计算向量的最小值，加各种参数按各种方式计算最小值.

tf.reduce_sum：对某一个维度内求和.  https://stackoverflow.com/questions/47157692/how-does-reduce-sum-work-in-tensorflow

tf.reduce_prod #沿维度相乘

tf.reduce_min #沿维度找最小

tf.reduce_max #沿维度找最大

tf.reduce_mean #沿维度求平均

tf.reduce_all  #沿维度与操作

tf.reduce_any  #沿维度或操作

tf.boolean_mask

```

- 两个tensor向量的加减乘除运算
```
tf.add

tf.argmax(vector, dimention)：返回的是vector中的最大值的索引号

tf.tensordot : https://stackoverflow.com/questions/41870228/understanding-tensordot

tf.multiply(x, y) 两个矩阵中对应元素各自相乘。要求x和y的形状必须一致。

tf.matmul(x, y) 将矩阵a乘以矩阵b，生成a * b。要求x的行数必须和y的列数相等。

注意以上两种乘法运算的区别。

tf.truediv 按元素除法x / y

```

- 两个tensor向量的concat操作
```python
tf.concat
在某个维度把两个tensor串联起来。

tf.sparse_concat
在某个维度把两个 sparse_concat 串联起来。
```

- 关于tensor向量的判断
```python
tf.equal

tf.where
tf.where(condition, x = None, y = None, name = None)，根据condition判定返回。即condition是True，选择x；condition是False，选择y。


tf.unique
换一种形式表达原来的向量。由原始向量变为 值向量 和 索引向量.
返回一个元组tuple(y,idx)，y为x的列表的唯一化数据列表，idx为x数据对应y元素的index
比如
tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx = unique(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]


tf.gather
用一个一维的索引数组,将张量中对应索引的值提取出来
比如 
b = tf.Variable([1,2,3,4,5,6,7,8,9,10])
index_b = tf.Variable([2,4,6,8])
那么 tf.gather(b, index_b) 的结果就是 [3 5 7 9]

```


### 向量标准化

tf.nn.l2_normalize

### 类型形式转换函数

https://www.tensorflow.org/api_guides/python/array_ops

```python
tf.cast

tf.expand_dims 增加一个维度，被增加的维度的数据长度就是1.

tf.reshape 

tf.squeeze 将原始input中所有维度为1的那些维都删掉

tf.tile 对当前张量内的数据进行一定规则的复制。最终的输出张量维度不变。
```

### 计算 Embedding 

```python
tf.nn.embedding_lookup

 # 加载总的词表
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
...
# 从总词表里查询当前输入的embedding
embed = tf.nn.embedding_lookup(embeddings, train_inputs) # lookup table 
```


tf.nn.embedding_lookup_sparse
```python
embedding_variable = tf.Variable(tf.truncated_normal([input_size, embedding_size], stddev=0.05), name='emb')
...
embedding = tf.nn.embedding_lookup_sparse(embedding_variable, sparse_id, sparse_value, "mod", combiner="sum")
```
embedding_variable 就是需要学习的参数，其中input_size是矩阵的行数，embedding_size是矩阵的列数，比如我们有100万个稀疏id，每个id要embedding到50维向量，那么矩阵的大小是[1000000, 50]。
sparse_id是要做向量化的一组id，用SparseTensor表示；sparse_value是每个id对应的一个value，用作权重，也用SparseTensor表示。

### tensorflow::Flag

用于解析和处理命令行参数



## 高阶函数

tf.map_fn



## 神经网络构建函数 Build Graph

### 网络层

tf.layers.dense / tf.layers.Dense

    inputs：输入该网络层的数据
    units：该层输出的维度大小，改变inputs的最后一维
    activation：激活函数，即神经网络的非线性变化
    use_bias：使用bias为True（默认使用），不用bias改成False即可，是否使用偏置项
    trainable=True:表明该层的参数是否参与训练。如果为真则变量加入到图集合中

tf.layers.batch_normalization

tf.layers.dropout

### 激活函数

```
tf.nn.relu
tf.nn.sigmoid
tf.nn.tanh
tf.nn.softmax

tf.nn.dropout
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

### 损失函数 Set Loss

- tf.nn.sigmoid_cross_entropy_with_logits
计算经sigmoid 函数激活之后的交叉熵.

- 交叉熵损失函数 tf.nn.softmax_cross_entropy_with_logits

- 交叉熵损失函数 tf.nn.sparse_softmax_cross_entropy_with_logits

上面两种交叉熵损失函数的区别：https://stackoverflow.com/questions/37312421/whats-the-difference-between-sparse-softmax-cross-entropy-with-logits-and-softm

- For sparse_softmax_cross_entropy_with_logits, labels must have the shape [batch_size] and the dtype int32 or int64. Each label is an int in range [0, num_classes-1].
- For softmax_cross_entropy_with_logits, labels must have the shape [batch_size, num_classes] and dtype float32 or float64.

softmax_cross_entropy_with_logits 是用的最多的，此外还有mean_squared_error和sigmoid_cross_entropy。

```
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_nn, labels=Y)

tf.reduce_mean(cross_entropy)
```

- KL散度，也就是两个分布的相对熵，体现的是两个分布的相似程度，熵越小越相似
```
tf.distributions.kl_divergence
```

### 优化器函数 Set Optimizer

最直接的优化方法自然是梯度下降：
```python
tf.gradients
tf.gradients(ys, xs)实现ys对xs求导

tf.stop_gradient
```


内置优化器有哪些？
```python
tf.train.AdamOptimizer
tf.train.GradientDescentOptimizer
```

优化器怎么用？
```python
my_opt = tf.train.GradientDescentOptimizer(0.02) # 参数时学习率
train_step = my_opt.minimize(loss) # 其中的loss是自己经过网络之后又构建好的损失值tensor
```

优化器函数是怎么更新整个网络参数的？
通过operation。 my_opt.minimize(loss)得到的就是一个op，把这个op传入到session.run(train_step)里面去，就会更新网络的权值。

```python
train_op = optim.minimize(loss, global_step=self.global_step, var_list=train_vars)
```
* var_list 参数指明了本次优化中可以被更新的权值。
* global_step 参数是训练迭代的计数器，比如说在Tensorboard画loss和 accuracy的横坐标即是global_step。优化器op每执行一次，该值就会自增1.
* gate_gradients  梯度计算的同步或异步


tf.GradientTape怎么用？
GradientTape是TF2引入的梯度计算方式，
```python
    def train_step(input, label):
        loss = 0.0
        with tf.GradientTape() as tape:
            probs = model(input)
            ...
            loss = loss_func(probs, label)

        batch_loss = loss
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss
```

## 参数控制函数

tf.train.exponential_decay

tf.train.Supervisor


## tf.slim

import tensorflow.contrib.slim as slim
slim是一个使构建，训练，评估神经网络变得简单的库。slim主要包括learning.py, evaluation.py, summary.py, queue.py和model_analyzer.py，分别对应模型的训练，测试，日志记录，队列管理和模型分析部分。


# 特征处理 Feature Columns

特征预处理是要将样本的 `原始数据` 变换为同模型适配的 `Tensor向量` 形式。

Feature Columns是Tensorflow中 原始数据 和 Estimator 的中间转换，这一过程是把换数据转换为适合Estimators使用的形式。机器学习模型用数值表示所有特征，而原始数据有数值型、类别型等各种表示形式。Feature Columns作为高阶API，其实就是在做特征预处理。

## 如何使用 Feature Columns？

feature_columns 作为 `Estimators的params参数`之一，它将输入数据 input_fn 和 模型 联系起来。
我们输入到`input_fn`中的训练数据也是依据feature_columns的格式生成的。可以看到 tf.feature_column 有很多种。

可参考 https://www.tensorflow.org/guide/feature_columns

- tf.feature_column.input_layer() 比较特殊，它作为输入层。

它的返回值是生产的dense Tensor，作为网络的输入层。

- tf.feature_column.make_parse_example_spec 方法将若干个feature_colunms转换为key-value字典形式（key是feature name， value是 FixedLenFeature 或 VarLenFeature）

```python
# Define features and transformations
feature_a = categorical_column_with_vocabulary_file(...)
feature_b = numeric_column(...)
feature_c_bucketized = bucketized_column(numeric_column("feature_c"), ...)
feature_a_x_feature_c = crossed_column(
    columns=["feature_a", feature_c_bucketized], ...)

feature_columns = set(
    [feature_b, feature_c_bucketized, feature_a_x_feature_c])

features = tf.parse_example(
    serialized=serialized_examples,
    features=make_parse_example_spec(feature_columns))

dense_tensor = tf.feature_column.input_layer(features, feature_columns)
```


## Numeric column
tf.feature_column.numeric_column

## Bucketized column
tf.feature_column.bucketized_column
将数据按范围切分为bucket。

## Categorical identity column
tf.feature_column.categorical_column_with_identity

## Categorical vocabulary column
tf.feature_column.categorical_column_with_vocabulary_list
tf.feature_column.categorical_column_with_vocabulary_file

## Hashed column
tf.feature_column.categorical_column_with_hash_bucket

## Crossed column
tf.feature_column.crossed_column

## Indicator column
tf.feature_column.indicator_column
对类型特征进行one-hot编码后的特征。
它是以Categorical column为输入基础的。

## Embedding column
tf.feature_column.embedding_column
对类型特征进行Embedding编码后的特征。
它是以Categorical column为输入基础的。

## tf.feature_column.shared_embedding_columns
若干个embedding column共享一模一样的权重数值。

## tf.feature_column.weighted_categorical_column
Applies weight values to a CategoricalColumn







# 样本文件/数据的格式化处理

我们需要关心，预测阶段的特征数据和训练阶段的样本数据，是以怎么样的形式进入到模型结构的Tensor当中的。 

## place_holder and feed_dict


tf.Example messages to and from tfrecord files

## 数据输入流
样本输入的过程称作是ETL过程，这一过程由Extract、Transform、Load三个步骤组成。
- Extract是从硬盘or网络磁盘到内存的过程。
- Transform是在内存中进行格式转换，比如从 protobuf 到 tf.data.Dataset.
- Load是将batch规模的样本加载到GPU加速设备上.

## tf.Example

TFRecord是文件形态， 而tf.Example / tf.train.example 是内存对象形态.

tf.Example is a {"string": tf.train.Feature} mapping.

- tf.train.Feature

tensorflow的 example 包含的是基于key-value对的存储方法，其中key是一个字符串，其映射到的是feature信息，feature包含三种类型：
		BytesList：字符串列表
		FloatList：浮点数列表
		Int64List：64位整数列表


### tf.train.SequenceExample

### tf.parse_example 方法

parse_example 方法把 序列化的特征(tf.Example) 解析为 字典类型(tensor)。
参考 tensorflow/python/ops/parsing_ops.py.
parse_example 的输入：
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos. 这是原始数据。
    features: A mapping dict from keys to `VarLenFeature`, `SparseFeature`, and `FixedLenFeature` objects. 这是期望解析成的schema。

1. VarlenFeature： 是按照键值把example的value映射到SpareTensor对象.
2. FixedLenFeature：是按照键值对将features映射到大小为 [serilized.size(), df.shape] 的Tensor矩阵.
3. SparseFeature：稀疏表示方式的feature，不推荐使用。

parse_example的输出：
    return: A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.


### tf.parse_single_example 方法

区别于tf.parse_example，tf.parse_single_example 只是少了一个batch而已，其余的都是一样的


## TFRecord

TFRecord是Tensorflow特有的二进制数据存储格式。它的好处是性能，在加载和传输时代价较小。另一个好处是可以存储序列化数据。

我们用Tensorflow API可以方便的构建和读写TFRecord数据。

tf.python_io.TFRecordWriter

使用tf_record_iterator方法可以从tfrecord文件中解析出json(k-v)形式的特征数据。
```python
# 这一方法已经被 deprecated
import tensorflow as tf
target_file = "tf_record_file_000"
for example in tf.python_io.tf_record_iterator(target_file):
    result = tf.train.Example.FromString(example)
```

## tf.data.Dataset

tf.data.Dataset 协助我们完成数据从文件形式到灌入Tensor的处理过程。
第一步生成Dataset，第二步生成Iterator，第三部循环获取Tensor。

在训练模型的时候，tf.data.Dataset 可以作为 input_fn 方法的返回值数据.
在进行预测的时候，tf.data.Dataset 

tf.data.Dataset.from_tensor_slices
tf.data.make_initializable_iterator(dataset)

下面的七行代码，我们使用tf.data.Dataset（不管是内置Dataset还是Custom Dataset）来完成ETL三个过程。
```python
with tf.name_scope("tf_record_reader"):
    # 1.Extract
    # generate file list
    files = tf.data.Dataset.list_files(glob_pattern, shuffle=training)

    # parallel fetch tfrecords dataset using the file list in parallel
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        lambda filename: tf.data.TFRecordDataset(filename), cycle_length=threads))

    # shuffle and repeat examples for better randomness and allow training beyond one epoch
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(32*self.batch_size))

    # 2.Transform
    # map the parse function to each example individually in threads*2 parallel calls
    dataset = dataset.map(map_func=lambda example: _parse_function(example, self.image_size, self.num_classes,training=training),
                          num_parallel_calls=threads)

    # batch the examples
    dataset = dataset.batch(batch_size=self.batch_size)

    # 3.Load
    #prefetch batch
    dataset = dataset.prefetch(buffer_size=self.batch_size)

    return dataset.make_one_shot_iterator()
```

```python
  with tf.Session() as sess:
    iterator = MyReaderDataset().make_one_shot_iterator()
    next_element = iterator.get_next()
    try:
      while True:
        print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
      pass
```

1. Dataset的map方法
This transformation applies map_func to each element of this dataset, and returns a new dataset containing the transformed elements, in the same order as they appeared in the input. 其中的 num_parallel_calls 参数可以指定并行处理的并发数。
```
map(
    map_func,
    num_parallel_calls=None
)
```
2. Dataset的apply方法
apply方法和map方法是什么区别？https://stackoverflow.com/questions/47091726/difference-between-tf-data-dataset-map-and-tf-data-dataset-apply

3. Dataset的interleave方法


## 自定义Dataset

设计自定义文件格式和自己的方法构建tensor，需要自己实现两个任务：
1. 文件格式：使用 tf.data.Dataset 阅读器来从文件中读取原始记录（通常以零阶字符串张量（scalar string tensors）表示，也可能有其他结构）。
2. 记录格式：使用解码器或者解析操作将一个字符串记录转换成 TensorFlow 可用的张量（tensor）。

以下是一种特别特殊的custom OP。

- DatasetOpKernel 的子类

要自己实现一个 tensorflow::DatasetOpKernel 的子类，这个类的 MakeDataset() 方法要告诉 TensorFlow 怎样根据一个操作的输入和属性生成一个数据集的对象。

- MakeDataset 方法要返回一个 DatasetBase 的子类

要自己实现 DatasetBase 的子类，这个类的 MakeIteratorInternal() 方法要构建`迭代器对象`。

- DatasetIterator 的子类

一个 tensorflow::DatasetIterator<Dataset> 的子类，表示特定数据集上的迭代器的可变性，这个类的 GetNextInternal() 方法告诉 TensorFlow 怎样获取迭代器的下一个元素。

GetNextInternal 定义了怎样从文件中实际读取记录，并用一个或多个 Tensor 对象来表示它们.

GetNextInternal 可能会被并发调用，所以推荐用一个互斥量来保护迭代器的状态。

```sh
    EnsureRunnerThreadStarted

      RunnerThread  通过StartThread开启的线程函数
        CallFunction
          map_func

      ProcessResult

    CallCompleted 释放锁
```

## 缺失值的处理

每一种Feature Column都有一项default_value作为输入参数。





# 模型文件格式

1. GraphDef
2. SavedModel

下面两种模型文件格式对应着tensorflow的两种模型文件保存方式。

checkpoint文件 是用于本地加载模型然后进行本地预测的。
pb-variable文件是用来让tensorflow serving加载并进行远程预测的。

在模型文件中（不管是何种格式），我们想保存的信息有两种：
1. a graph (various operations).
2. weights/variables in a graph.

谷歌推荐的保存模型的方式是保存模型为 PB 文件，它具有语言独立性，可独立运行，封闭的序列化格式，任何语言都可以解析它，它允许其他语言和深度学习框架读取、继续训练和迁移 TensorFlow 的模型。

## checkpoint文件

这是由 tf.train.Saver 类生成的模型文件。

The .ckpt is the model given by tensorflow which includes all the 
weights/parameters in the model.

checkpoints，包含三个主要文件，meta, index, data。
meta主要有各种def，一个很重要的就是graph_def，而data保存真正的weight。

checkpoints, which are versions of the model created during training. 存储的为最近的几次迭代保存的模型名称以及路径：

    meta file: 在meta文件中保存的是模型的图。describes the saved graph structure, includes GraphDef, SaverDef, and so on; then apply tf.train.import_meta_graph('/tmp/model.ckpt.meta'), will restore Saver and Graph.
	
		index file: 在index文件中保存的为模型参数的名称以及具体属性。it is a string-string immutable table(tensorflow::table::Table). Each key is a name of a tensor and its value is a serialized BundleEntryProto. Each BundleEntryProto describes the metadata of a tensor: which of the "data" files contains the content of a tensor, the offset into that file, checksum, some auxiliary data, etc.
	
		data file: 在data文件中保存的为模型参数的数值。it is TensorBundle collection, save the values of all variables.

https://www.tensorflow.org/guide/checkpoints


## pb-variable文件(SavedModel)

这是由 tf.saved_model.builder.SavedModelBuilder 类生成的模型文件。

总的来说，variables保存所有变量; saved_model.pb用于保存模型结构等信息。

- saved_model.pb文件，其实就是graph_def，但是指的一般是做了constant化，这样可以直接加载做inference，安装部署用。
The .pb file stores the computational graph. Includes the graph definitions as `MetaGraphDef` protocol buffers.
PB是表示 MetaGraph 的 protocol buffer格式的文件，MetaGraph 包括计算图，数据流，以及相关的变量和输入输出signature以及 asserts 指创建计算图时额外的文件。

- variables.data:保存的是模型结构图中的op和参数变量之间的对应关系

- variables.index:保存的是变量值

- assets目录文件：我们常说的模型词表文件，就放在这个目录下。 assets is a subfolder containing auxiliary (external) files, such as vocabularies. Assets are copied to the SavedModel location and can be read when loading a specific MetaGraphDef.

```
	|-- mnist_saved_model
	|   `-- 1531711208
	|       |-- saved_model.pb   保存了serialized tensorflow::SavedModel
  |       |-- assets 
	|       `-- variables   保存了variables
	|           |-- variables.data-00000-of-00001
	|           `-- variables.index
```

tf.train.Saver也能导出variables文件。

### pb/pbtxt中的信息

其实这里要说的就是 message SavedModel （tensorflow/core/protobuf/saved_model.proto）的定义。
官方文档：https://www.tensorflow.org/guide/extend/model_files
SavedModel的核心元素是 message MetaGraphDef 。下面六类message是 MetaGraphDef 的核心元素。

- MetaInfoDef meta_info_def
```
    stripped_op_list {} # 这里罗列了图中所有的OP的名称、输入输出名称属性、属性
    tags: "serve"
    tensorflow_version: "1.13.1"
    tensorflow_git_version: "b\'unknown\'"
    stripped_default_attrs: true
```
- GraphDef graph_def
由大量的 NodeDef 而组成的有向图（计算图）。每一个NodeDef的名称、op名、输入名、输出形态
- SaverDef saver_def
configuration of a Saver.
```
    filename_tensor_name: "save/Const:0"
    save_tensor_name: "save/Identity:0"
    restore_op_name: "save/restore_all"
    max_to_keep: 5
    sharded: true
    keep_checkpoint_every_n_hours: 10000.0
    version: V2
```
- CollectionDef collection_def
```
    saved_model_main_op
    table_initializer
    train_op
    trainable_variables
    variables
```
- SignatureDef signature_def
SignatureDef的作用是定义 输出 和 输入接口。
- AssetFileDef asset_file_def
Asset file def to be used with the defined graph.
- SavedObjectGraph object_graph_def
Extra information about the structure of functions and stateful objects.

### 图中的op都是怎么样的op？ 是否包含custom op？ 哪些高阶OP是直接体现在图中的？哪些高阶OP是以简单OP的组合体现在图中的？
通过Python函数 export_savedmodel 导出生成的图中，包含的全部都是最原始的op操作，一些高阶的py操作都会转换为原始op。


### pb 和 pbtxt 之间的转换

There are actually two different formats that a ProtoBuf can be saved in. 

两种格式转换可以通过 tensorflow.core.protobuf.saved_model_pb2

```python
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
def pb_to_pbtxt(pbtxt_filename, pb_filename):
    with gfile.FastGFile(pb_filename, 'rb') as f:
        saved_model = saved_model_pb2.SavedModel()
        saved_model.ParseFromString(f.read())
        with open(pbtxt_filename, 'w') as g:
            g.write(str(saved_model))
```

```python
# pb 2 pbtxt
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
def graphdef_to_pbtxt(filename): 
  with  tf.compat.v1.gfile.FastGFile(filename, 'rb') as f:
    data = tf.compat.as_bytes(f.read())
    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString(data)
  with open('saved_model.pbtxt', 'w') as fp:
    fp.write(str(sm))
graphdef_to_pbtxt('saved_model.pb')
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

下面是构建serving pb-variable文件的过程：
```py
tf.saved_model.builder.SavedModelBuilder

tf.saved_model.utils.build_tensor_info

tf.saved_model.signature_def_utils.build_signature_def

builder.add_meta_graph_and_variables

builder.save()
```

## 模型保存/导出和加载函数

我们经常在训练完一个模型之后希望保存训练的结果，这些结果指的是模型的参数，以便下次迭代的训练、或用作测试、或用于预测。

1. 第一种：是传统的 tf.train.Saver 类save保存和restore恢复方法。Tensorflow针对这一需求提供了Saver类。
这种方法将模型保存为ckpt格式。

tf.train.get_checkpoint_state   输入路径必须是绝对路径

```python
# 保存
saver = tf.train.Saver() #什么参数都不输入，则保存all saveable objects，存储形式为ckpt
save_path = saver.save(sess, model_path) 
...
saver = tf.train.Saver({"embeddings": embeddings}) #输入包含variable的字典，存储形式为variables
saver.save(sess, "./lu_vari")
...

# 加载
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
# tf.train.latest_checkpoint自动获取最后一次保存的模型
saver.restore(sess, model_path)
```

2. 第二种：是比较新颖的 tf.saved_model.builder.SavedModelBuilder 类的builder保存和loader文件里的load恢复方法。
这种方法将模型保存为pb-variable格式。

```python
# 保存
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
builder.add_meta_graph_and_variables(...)
builder.save()
...
# 加载
tf.saved_model.loader.load(sess, ["tag"], export_dir)
```

3. 第三种：高阶API版的方法
tf.estimator.Estimator.export_savedmodel
这种方法将模型也保存为pb-variable格式。

```python
export_savedmodel(
    export_dir_base,
    serving_input_receiver_fn,
    assets_extra=None,
    as_text=False,
    checkpoint_path=None,
    strip_default_attrs=False
)
```


导出用于Serving的模型
一般要去除仅用于Training的op节点。
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py



## checkpoint文件 和 pb-variable文件之间的转换



## 如何可视化展示导出模型文件里的图？

https://github.com/tensorflow/tensorflow/issues/8854

下面这个脚本就是把输入的saved_model.pb文件转换为能够被tensorboard展示的 events 文件。
```python
import tensorflow as tf
import sys
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

with tf.Session() as sess:
	model_filename =sys.argv[1]
	with gfile.FastGFile(model_filename, 'rb') as f:

		data = compat.as_bytes(f.read())
		sm = saved_model_pb2.SavedModel()
		sm.ParseFromString(data)
		if 1 != len(sm.meta_graphs):
			print('More than one graph found. Not sure which to write')
			sys.exit(1)
			
		g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
	LOGDIR=sys.argv[2]
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()
```

## 导出模型文件中常见的OP

比较特别的OP比如 Placeholder ，用于放置input。

tf.gater

tf.where

tf.split

tf.concat

tf.expend_dim

tf.unsorted_segment_sum

tf.unstack

tf.stack

tf.slice

tf.erf

```
op: "Add"
op: "Assign"
op: "Cast"
op: "ConcatV2"
op: "Const"
op: "ExpandDims"
op: "Fill"
op: "GatherNd"
op: "GatherV2"
op: "Greater"
op: "GreaterEqual"
op: "HashTableV2"
op: "HistogramSummary"
op: "Identity"
op: "InitializeTableFromTextFileV2"
op: "LookupTableFindV2"
op: "MatMul"
op: "Maximum"
op: "MergeV2Checkpoints"
op: "Minimum"
op: "Mul"
op: "NoOp"
op: "NotEqual"
op: "Pack"
op: "ParseExample"
op: "Placeholder"
op: "Prod"
op: "RealDiv"
op: "Relu"
op: "Reshape"
op: "RestoreV2"
op: "Rsqrt"
op: "SaveV2"
op: "Select"
op: "Shape"
op: "ShardedFilename"
op: "Sigmoid"
op: "Slice"
op: "SparseFillEmptyRows"
op: "SparseReshape"
op: "SparseSegmentSqrtN"
op: "Square"
op: "StridedSlice"
op: "StringJoin"
op: "Sub"
op: "Sum"
op: "Tile"
op: "TokenizeSparse"
op: "TruncatedNormal"
op: "Unique"
op: "VariableV2"
op: "Where"
op: "ZerosLike"
```






# 模型训练方式

## Multi-output


## Multi-head / Multi-task / Muti-target DNN

比如把点击率和下单率作为两个目标，分别计算各自的loss function。DNN的前几层作为共享层，两个目标共享这几层的表达，在BP阶段根据两个目标算出的梯度共同进行参数更新。网络的最后用一个全连接层进行拆分，单独学习对应loss的参数。

### Multi-objective learning

tf.contrib.estimator.multi_head

在input_fn中设定多个label作为多个目标。
在model_fn中创建多种_head。

## Warm Start

Tensorflow 中有一个方法tf.estimator.WarmStartSettings。
tf.estimator.DNNClassifier 方法中有一个参数叫 warm_start_from。
https://www.tensorflow.org/api_docs/python/tf/estimator/WarmStartSettings

这个参数的作用是：Optional string filepath to a checkpoint or SavedModel to warm-start from, or a tf.estimator.WarmStartSettings object to fully configure warm-starting. If the string filepath is provided instead of a tf.estimator.WarmStartSettings, then all variables are warm-started, and it is assumed that vocabularies and tf.Tensor names are unchanged.


```python
emb_vocab_file = tf.feature_column.embedding_column(
    tf.feature_column.categorical_column_with_vocabulary_file(
        "sc_vocab_file", "new_vocab.txt", vocab_size=100),
    dimension=8)
emb_vocab_list = tf.feature_column.embedding_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        "sc_vocab_list", vocabulary_list=["a", "b"]),
    dimension=8)
estimator = tf.estimator.DNNClassifier(
  hidden_units=[128, 64], feature_columns=[emb_vocab_file, emb_vocab_list],
  warm_start_from=ws)
```
ws这个参数可以由tf.estimator.WarmStartSettings生成的。
ws也可以是指定的路径，用于加载checkpoint或者savedmodel文件。

tf.estimator.WarmStartSettings的参数：
		ckpt_to_initialize_from=warm_start_checkpoint_path, 指定checkpoint文件的位置
		var_name_to_vocab_info=var_infos,  表示WarmStartSettings 的词汇信息
		var_name_to_prev_var_name=config.var_name_to_prev_var_name



## 额外加载预训练的Embedding词表

```python
tf.train.load_variable

W = tf.get_variable(name="W", shape=embedding.shape, initializer=tf.constant_initializer(embedding), trainable=False)
```

## 超大规模稀疏参数 

### 借助recommenders-addons项目
为了支持TF上进行超大稀疏特征所对应的稀疏参数训练，针对于搜索、推荐、广告领域的稀疏模型引入了动态Embedding技术。
https://github.com/tensorflow/recommenders-addons

1. 原生TF的 tf.Variable 是固定长度，只能修改weight的值，而不支持动态新增和删除weight。
2. 稀疏参数如果以kv形式存储在hash map里，没法直接训练。

```python
features = tf.parse_example(...)
...
col0 = tf.contrib.layers.sparse_column_with_hash_bucket(column_name = "col0",
                                                        hash_bucket_size = 10000)
feature_columns = [tf.feature_column.embedding_column(categorical_column = col0,
                                                      dimension = 16)]
input_data = tf.contrib.layers.input_from_feature_columns(columns_to_tensors = features,
                                                          feature_columns = feature_columns)
...
```

### 借助阿里的DeepRec项目
https://github.com/alibaba/DeepRec
* 使用get_embedding_variable接口
* 使用categorical_column_with_embedding接口
* 进阶功能： 特征淘汰、特征准入、动态维度


## Transfer Learning - Retrained Model

通过迁移学习，我们不需要太多的数据！这个想法是从一个以前在数百万图像上训练过的网络开始的，比如在ImageNet上预训练的ResNet。然后，我们将通过仅重新训练最后几个层并使其他层独立来微调ResNet模型。

通常我们会把已有模型中前面的几层直接拿过来用，重新设计后面几层的结构，重新训练后面几层的权值，注意新训练的时候要把前面几层的权值固定。

bottleneck指的是网络最后输出层之前的一层（倒数第二层）。这一层中原始的特征已经经过了前面若干层而被压缩到了新的表示空间。对于图像分类网络来讲，它就是image feature vector。

### bottleneck layer - penultimate layer


### final layer retraining


### Fine Tune
Fine Tune通常指的就是冻结网络前面的层，然后训练最后一层。

在调用优化器的 minimize 方法生成训练op的时候，可以传入一个参数var_list来指定可以被优化的参数。
```python
output_vars = tf.get_collection(tf.GraphKyes.TRAINABLE_VARIABLES, scope='outpt')
train_step = optimizer.minimize(loss_score,var_list = output_vars)
```

### retrain一个model

```python
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/mobilenet_0.50_224 \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture=mobilenet_0.50_224 \
  --image_dir=tf_files/flower_photos
```

jpeg_data_tensor -> decoded_image_tensor -> resized_input_tensor



### 使用retrained model进行预测

```python
python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --labels=tf_files/retrained_labels.txt  \
    --image=tf_files/flower_photos/roses/2414954629_3708a1a04d.jpg 
```











# TensorFlow Serving

TensorFlow Serving 是基于 gRPC 和 Protocol Buffers 开发的。
https://github.com/tensorflow/serving ,
https://www.tensorflow.org/serving/serving_basic

## 服务端 tensorflow-model-server

在服务器端安装好之后，核心就是 tensorflow_model_server 这个binary。

		tensorflow_model_server --help

可以通过apt-get直接安装官方版本的tensorflow_model_server。当然也可以自己通过bazel编译出 tensorflow_model_server

    CC=gcc-4.9 CXX=g++-4.9  bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server --jobs 2 --local_resources 2048,3.0,1.0

    CC=gcc-4.9 CXX=g++-4.9  bazel build -c opt tensorflow_serving/... --jobs 2 --local_resources 2048,3.0,1.0

    编译之后，可执行文件 tensorflow_model_server 就放在 /tensorflow-serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server

如果编译环境的内存不够大或者gcc版本过高，编译时就容易遇到编译系统错误。所以我编译的时候主动用的较低的gcc版本4.9来编译。

1. 在服务端先要训练一个模型

可以用 models repo 中的例子：

		cd models/official/mnist
		python mnist.py --export_dir ./tmp/mnist_saved_model

或者用 tensorflow_serving repo中的例子：

		cd tensorflow_serving/example/
		python mnist_saved_model.py ./tmp/mnist_model

2. 保存的模型是这样子的：

```shell
	|-- mnist_saved_model
	|   `-- 1531711208
	|       |-- saved_model.pb   保存了serialized tensorflow::SavedModel.
	|       `-- variables   保存了variables.
	|           |-- variables.data-00000-of-00001
	|           `-- variables.index
```

源代码下有一些模型例子 tensorflow_serving/servables/tensorflow/testdata

3. 然后将这个模型载入到 TensorFlow ModelServer，注意输入的模型路径必须是绝对路径。

    tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/

    /tensorflow-serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
      --port=8700 \
      --model_name=half_plus_two \
      --model_base_path=/tensorflow-serving/tensorflow_serving/servables/tensorflow/testdata/half_plus_two/

或者使用配置文件加载模型

    tensorflow_model_server --port=9000 --model_config_file=/serving/models.conf


1. gRPC方式对外提供服务

默认使用 --port方式就是以gPRC方式提供服务。

```shell
$ tensorflow_model_server \
      --port=9000 \
      --model_name=mnist \
      --model_base_path=/tmp/mnist_model/
```

5. RESTful方式对外提供服务

用参数指明要使用rest方式提供服务。
```shell
$ tensorflow_model_server \
   --rest_api_port=8501 \
   --model_name=half_plus_three \
   --model_base_path=$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_three/
```

6. 更多启动参数

```shell
usage: /tensorflow-serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
Flags:
	--port=8500
    int32	Port to listen on for gRPC API
	--grpc_socket_path=""
    string	If non-empty, listen to a UNIX socket for gRPC API on the given path. Can be either relative or absolute path.
	--rest_api_port=0                	
    int32	Port to listen on for HTTP/REST API. If set to zero HTTP/REST API will not be exported. This port must be different than the one specified in --port.
	--rest_api_num_threads=16        	
    int32	Number of threads for HTTP/REST API processing. If not set, will be auto set based on number of CPUs.
	--rest_api_timeout_in_ms=30000   	
    int32	Timeout for HTTP/REST API calls.
	--enable_batching=false          	
    bool	enable batching
	--allow_version_labels_for_unavailable_models=false	bool	
    If true, allows assigning unused version labels to models that are not available yet.
	--batching_parameters_file=""    	
    string	If non-empty, read an ascii BatchingParameters protobuf from the supplied file name and use the contained values instead of the defaults.
	--model_config_file=""           	
    string	If non-empty, read an ascii ModelServerConfig protobuf from the supplied file name, and serve the models in that file. This config file can be used to specify multiple models to serve and other advanced parameters including non-default version policy. (If used, --model_name, --model_base_path are ignored.)
	--model_config_file_poll_wait_seconds=0	int32	Interval in seconds between each poll of the filesystemfor model_config_file. If unset or set to zero, poll will be done exactly once and not periodically. Setting this to negative is reserved for testing purposes only.
	--model_name="default"           	
    string	name of model (ignored if --model_config_file flag is set)
	--model_base_path=""             	
    string	path to export (ignored if --model_config_file flag is set, otherwise required)
	--max_num_load_retries=5         	
    int32	maximum number of times it retries loading a model after the first failure, before giving up. If set to 0, a load is attempted only once. Default: 5
	--load_retry_interval_micros=60000000	
    int64	The interval, in microseconds, between each servable load retry. If set negative, it doesnt wait. Default: 1 minute
	--file_system_poll_wait_seconds=1	
    int32	Interval in seconds between each poll of the filesystem for new model version. If set to zero poll will be exactly done once and not periodically. Setting this to negative value will disable polling entirely causing ModelServer to indefinitely wait for a new model at startup. Negative values are reserved for testing purposes only.
	--flush_filesystem_caches=true   	
    bool	If true (the default), filesystem caches will be flushed after the initial load of all servables, and after each subsequent individual servable reload (if the number of load threads is 1). This reduces memory consumption of the model server, at the potential cost of cache misses if model files are accessed after servables are loaded.
	--tensorflow_session_parallelism=0	
    int64	Number of threads to use for running a Tensorflow session. Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.
	--tensorflow_intra_op_parallelism=0	
    int64	Number of threads to use to parallelize the executionof an individual op. Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.
	--tensorflow_inter_op_parallelism=0	
    int64	Controls the number of operators that can be executed simultaneously. Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.
	--ssl_config_file=""             	
    string	If non-empty, read an ascii SSLConfig protobuf from the supplied file name and set up a secure gRPC channel
	--platform_config_file=""        	
    string	If non-empty, read an ascii PlatformConfigMap protobuf from the supplied file name, and use that platform config instead of the Tensorflow platform. (If used, --enable_batching is ignored.)
	--per_process_gpu_memory_fraction=0.000000	
    float	Fraction that each process occupies of the GPU memory space the value is between 0.0 and 1.0 (with 0.0 as the default) If 1.0, the server will allocate all the memory when the server starts, If 0.0, Tensorflow will automatically select a value.
	--saved_model_tags="serve"       	
    string	Comma-separated set of tags corresponding to the meta graph def to load from SavedModel.
	--grpc_channel_arguments=""      	
    string	A comma separated list of arguments to be passed to the grpc server. (e.g. grpc.max_connection_age_ms=2000)
	--enable_model_warmup=true       	
    bool	Enables model warmup, which triggers lazy initializations (such as TF optimizations) at load time, to reduce first request latency.
	--version=false                  	
    bool	Display version
	--monitoring_config_file=""      	
    string	If non-empty, read an ascii MonitoringConfig protobuf from the supplied file name
	--remove_unused_fields_from_bundle_metagraph=true	bool	Removes unused fields from MetaGraphDef proto message to save memory.
	--use_tflite_model=false         	
    bool	EXPERIMENTAL; CAN BE REMOVED ANYTIME! Load and use TensorFlow Lite model from `model.tflite` file in SavedModel directory instead of the TensorFlow model from `saved_model.pb` file.
```


## 客户端 tensorflow-serving-api

在客户端把样本数据作为请求发送到TensorFlow ModelServer，

		python tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=localhost:9000

通过http直接请求

    curl -d '{"instances": [1.0, 2.0, 5.0]}'  -X POST http://localhost:8501/v1/models/half_plus_two:predict

## Client-Server 交互过程

具体的交互流程一般是这样：
下面的代码中每次请求可能传入多条样本（多条预测请求）。
```cpp
// TFS request 伪代码
tensorflow::Example example;
tensorflow::Features* features = example.mutable_features(); // tensorflow::Features内部是一个map
google::protobuf::Map<std::string, tensorflow::Feature >& feature_map = *features->mutable_feature();
for (auto y: DOC_LIST) { // 待预测的样本条数
    for (auto x: FEATUTES_LIST) { // 特征个数
        float value = 3.1415926; // get feature_value
        std::string fname = "a_feature_name"; // get feature_name
        tensorflow::Feature _feature; // create a Feature
        tensorflow::FloatList* fl = _feature.mutable_float_list(); //支持bytes_list、float_list、int64_list
        fl->add_value(value);
        feature_map[fname] = _feature;
    }
    std::string example_str = "";
    example.SerializeToString(&example_str); // 把整个tensorflow::Example序列化成string
    tensorflow::TensorProto example_proto;
    example_proto.set_dtype(tensorflow::DataType::DT_STRING);
    example_proto.add_string_val(example_str); // 把序列化string放入tensorflow::TensorProto
}
example_proto.mutable_tensor_shape()->add_dim()->set_size(PREDICT_CNT);
tensorflow::serving::PredictRequest* request_pb = static_cast<tensorflow::serving::PredictRequest*>(request);
request_pb->mutable_model_spec()->set_name("model-name");
request_pb->mutable_model_spec()->set_version_label("model-version");
google::protobuf::Map<std::string, tensorflow::TensorProto>& inputs = *request_pb->mutable_inputs();
inputs["examples"] = example_proto; // 这个例子中只在tensorflow::serving::PredictRequest中放入了一个k-v
```

下面的代码中一次读取请求时传入的多条样本所对应的结果。
```cpp
// TFS response 伪代码
tensorflow::serving::PredictResponse* response_pb = static_cast<tensorflow::serving::PredictResponse*>(response);
const google::protobuf::Map<std::string, tensorflow::TensorProto>& map_outputs = response_pb->outputs();
std::vector<float> scores;
for (auto x: map_outputs) {
    // x.first
    // x.second
    tensorflow::TensorProto& result_tensor_proto = x.second;
    for (int i = 0; i < result_tensor_proto.float_val_size(); i++) {
        scores.push_back(result_tensor_proto.float_val(i));
    }
}
```


## TensorFlow Serving 客户端-服务端数据交互格式

- Feature.proto 和 example.proto
定义在tensorflow/core/example/feature.proto 和 tensorflow/core/example/example.proto


- TensorProto
TensorProto是一个pb message，定义在tensorflow/core/framework/tensor.proto，用来表示一个Tensor。

- TensorInfo


- SignatureDef
由inputs TensorInfo、outputs TensorInfo、method_name三个成员构成。
SignatureDef的主要作用是定义输出和输入接口协议
A SignatureDef defines the signature of a computation supported in a TensorFlow graph. SignatureDefs aim to provide generic support to identify inputs and outputs of a function and can be specified when building a SavedModel.
```
message SignatureDef {
  map<string, TensorInfo> inputs = 1;
  map<string, TensorInfo> outputs = 2;
  string method_name = 3;
}
```

- SignatureDefMap
由 name->SignatureDef 构成的map。

- MetaGraphDef
由一个计算图 GraphDef 和其相关的元数据（SignatureDef、CollectionDef、SaverDef）构成。其包含了用于继续训练，实施评估和（在已训练好的的图上）做前向推断的信息。
定义在tensorflow/core/framework/graph.proto


- PredictRequest
由 map<string, TensorProto> 作为请求输入。要预测的样本就放在其中。

- PredictResponse
由 map<string, TensorProto> 作为请求返回。预测的结果就放在其中。


## input receiver 解析输入

对于Serving来说，预测时的输入data就是 `tf.example` 形式的.

serving_input_receiver_fn 方法在serving阶段，相当于训练阶段的 input_fn 方法。

- 它返回了一个 ServingInputReceiver 对象。 这个对象创建时传入了两个参数：
  一个是 receiver_tensors={receiver_key: serialized_tf_example}.
  一个是 features=parsing_ops.parse_example(serialized_tf_example,
                                           feature_spec)，它定义了传给模型的features.
  它是要把 tf.Example 解析为 tensor. 

  其中 serialized_tf_example 是 新创建的tf.placeholder，把它设计在要导出的graph中，用于接收input数据。
  其中 feature_spec 就相当于训练时设定的feature_columns。

serving_input_receiver_fn 是在导出模型时被使用的。
在导出模型的时候，会将 serving_input_receiver_fn 方法传入到 export_savedmodel 方法中。


tf.estimator.export.ServingInputReceiver 和 tf.estimator.export.TensorServingInputReceiver 有一点点差异：
`tf.estimator.export.TensorServingInputReceiver` allows `tf.estimator.Estimator.export_savedmodel` to pass raw tensors to model functions. TensorServingInputReceiver是一种特殊形式，只有一个tensor作为输入。


## Serving 内部是 怎么加载模型 和 怎么做预测的？

`Servables` 是一个抽象对象，它是serving对一个模型的表示，它指的是提供给客户端的一种计算。
一个典型的 `Servables` 包含：
  一个TensorFlow SavedModelBundle (`tensorflow::Session`)
  一个lookup table（查embedding或者vocabulary）。


SavedModelBundle 是核心模块，它要将来自指定文件的模型表示回graph，提供像训练时那样的Session::Run方法来做预测。
SavedModelBundle 结构中保存了 Session 指针和 MetaGraphDef。

Serve request with TensorFlow Serving `ServerCore`.


## 模型热加载 Runtime Reload Model

https://github.com/tensorflow/serving/issues/380

https://github.com/tensorflow/serving/issues/678

模型管理和模型热加载是由 TensorFlow Serving Manager 负责。

ServerCore::Create做了几件重要的事情：

- Instantiates a FileSystemStoragePathSource that monitors model export paths declared in model_config_list.
- Instantiates a SourceAdapter using the PlatformConfigMap with the model platform declared in model_config_list and connects the FileSystemStoragePathSource to it. This way, whenever a new model version is discovered under the export path, the SavedModelBundleSourceAdapter adapts it to a Loader<SavedModelBundle>.
- Instantiates a specific implementation of Manager called AspiredVersionsManager that manages all such Loader instances created by the SavedModelBundleSourceAdapter. ServerCore exports the Manager interface by delegating the calls to AspiredVersionsManager.




## 服务多个模型

使用配置文件的形式加载模型。
https://www.tensorflow.org/serving/serving_config

```
model_config_list: {
  config: {
    name: "deep_ranking",
    base_path: "/root/tensorflows/model/deep_ranking",
    model_platform: "tensorflow"
    model_version_policy: {
      latest: {
        num_versions: 8
      }
    }
  },
  config: {
    name: "pdnn_model",
    base_path: "/root/tensorflows/model/dnn_model",
    model_platform: "tensorflow"
    model_version_policy: {
      specific {
        versions: 42
        versions: 43
      }
    }
  }
}
```


## tensorflow serving with custom_op

使用一个模型进行预测，除了需要模型文件，需要特定形式的预测特征输入，还需要什么吗？还需要custom ops。

https://github.com/tensorflow/custom-op/issues/3

You can add your op/kernel BUILD targets to the list of SUPPORTED_TENSORFLOW_OPS and recompile the ModelServer.

1. adding my custom op in tensorflow_serving/model_servers/BUILD

```
SUPPORTED_TENSORFLOW_OPS = [
    "@org_tensorflow//tensorflow/contrib:contrib_kernels",
    "@org_tensorflow//tensorflow/contrib:contrib_ops_op_lib",
    "//tensorflow_serving/myop:myOp.so" #Added this line
]
```

2. add cc_library & tf_custom_op_library in tensorflow/core/custom_ops/BUILD


## Custom Servable

前面说了 Servable 是提供给client的一个计算。最典型的servable是 SavedModelBundle.

如何实现一个自己的 servable？ 官方提供了一种规范。

1. 创建 SourceAdapter 子类
1.1 实现 XX_source_adapter.h XX_source_adapter.cc
这个类要继承 SimpleLoaderSourceAdapter
1.2 定义 XX_source_adapter.proto

1.2 注册 source adapter
REGISTER_STORAGE_PATH_SOURCE_ADAPTER

2. 创建 Loader 子类










# Tensorflow 性能调优（训练/预测） 

- 服务器端的模型最在意的延时和吞吐率。
- 本地端的模型最在意的是CPU资源占用率、内存占用率。

## Benchmarks

https://github.com/tensorflow/benchmarks

## 性能分析模块 TensorFlow Profiler

https://www.tensorflow.org/guide/profiler
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/options.md
https://github.com/tensorflow/tensorflow/tree/r1.15/tensorflow/core/profiler
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler

### TF Trace / tf.RunMetadata / timeline对象
这是低阶API才能使用使用的方法。
使用 run_metadata 将每次session run的性能信息记录下来。
Timeline类可以被用于以Chrome Tracing的格式生成一个JSON trace文件。
生成的trace文件可以用 chrome://tracing/ 直接打开显示。
```python
from tensorflow.python.client import timeline
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
_ = sess.run(optimizer, options=options, run_metadata=run_metadata)

fetched_timeline = timeline.Timeline(run_metadata.step_stats)
chrome_trace = fetched_timeline.generate_chrome_trace_format()
with open(FLAGS.trace_file, 'w') as f:
    f.write(chrome_trace)
print('Chrome Trace File write in %s' % FLAGS.trace_file)
```
Python版本通过generate_chrome_trace_format方法能够生产一个json文件，以traceEvents为最外层的key。直接能用chrome浏览器打开。

```cpp
std::vector<Tensor> outputs;
RunOptions run_options;
RunMetadata run_metadata;
run_options.set_trace_level(RunOptions::FULL_TRACE);
session->Run(run_options, inputs, output_tensor_names, {},
                                  &outputs, &run_metadata);

std::string outfile = "serialized";
run_metadata.step_stats().SerializeToString(&outfile);
std::ofstream ofs("perf_trace");
ofs << outfile;
ofs.close();
```
C++版本是直接生成了一个二进制的traceEvent文件，可以通过独立的py来将装成
参考： https://github.com/tensorflow/tensorflow/issues/21312

```python
# StepStats2Timeline.py
from tensorflow.core.framework.step_stats_pb2 import StepStats
from tensorflow.python.client import timeline

f1 = open("perf_trace")
serialized = f1.read()
step_stats = StepStats()
step_stats.ParseFromString(serialized)

fetched_timeline = timeline.Timeline(step_stats)
chrome_trace = fetched_timeline.generate_chrome_trace_format()
with open("perf_trace.json", 'w') as f:
    f.write(chrome_trace)

```

### Timeline文件解析

hardware tracing 和 software tracing

MemcpyHtoD 和 MemcpyDtoH 是host和device之间拷贝内存，host指的是cpu主机。


一个CPU模型生成的timeline
```shell
/host:CPU Compute (pid 1)

/job:localhost/replica:0/task:0/device:CPU:0 Compute (pid 3)   - Fully qualified name 
```

一个GPU模型生成的timeline
```shell
/device:GPU:0/stream:all Compute (pid 7)

/gpu:0 (Tesla P40)/context#0/stream#1 Compute (pid 5)  - Short-hand notation

/gpu:0 (Tesla P40)/context#0/stream#2:MemcpyDtoH Compute (pid 3)

/gpu:0 (Tesla P40)/context#0/stream#3:MemcpyHtoD Compute (pid 1)

/host:CPU Compute (pid 9)

/job:localhost/replica:0/task:0/device:CPU:0 Compute (pid 11)

/job:localhost/replica:0/task:0/device:GPU:0 Compute (pid 13)
```


### tfprof
tf.contrib.tfprof.ProfileContext
```python
with tf.contrib.tfprof.ProfileContext(args.profile_dir) as pctx:
  run... # 可以是高阶API、也可以是低阶API
```

### tf.train.ProfilerHook
```python
hook = tf.train.ProfilerHook(save_steps=100, output_dir='/tmp/')
estimator.train(
      input_fn=lambda: ltr_dataset.csv_input_fn(train_file_list, args.batch_size),
      hooks=[hook]
)
```





## Grappler 模块

runtime图优化: https://www.tensorflow.org/guide/graph_optimization

Grappler是优化模块，包括：
  - tensorflow.gappler.ModelPruner 裁剪图中不需要的节点
  - tensorflow.grappler.ConstantFolding 做常量的折叠，所谓的常量折叠是将计算图中可以预先可以确定输出值的节点替换成常量，并对计算图进行一些结构简化的操作。
  - tensorflow.grappler.LayoutOptimizer类的主要优化逻辑是改变一些运算节点的输入数据格式来提高运算效率。
  - tensorflow.grappler.MemoryOptimizer 把一些计算中间结果交换到其他内存，需要时再换回，以节省计算设备的内存占用。
  - tensorflow.grappler.AutoParallel的优化逻辑是通过重构原来的计算图，使得模型的训练过程实现数据并行，准确的说是多个batch的数据能并行训练，而不用等前一个batch训练完成。

## XLA

XLA是将tensorflow.GraphDef编译成可执行代码。

XLA提供了AOT(提前编译)和JIT(即时编译)两种方式:
- AOT(提前编译)方式就是在代码执行阶段之前全部编译成目标指令，进入执行阶段后，不再有编译过程发生。
- JIT全称Just In Time（即时）.在即时编译中，计算图在不会在运行阶段前被编译成可执行代码，而是在进入运行阶段后的适当的时机才会被编译成可执行代码，并且可以被直接调用了。

在创建 Session 时，增加 config 参数，设置 config.graph_options.optimizer_options.global_jit_level 值为 tf.OptimizerOptions.ON_1 即可打开 XLA JIT 功能。注：该配置对整个 Session 生效，所有 OP 都会受到影响。(https://mp.weixin.qq.com/s/tBb2_X-lQvW-7puWS4XrlQ)

```python
  config = tf.ConfigProto()
  config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  sess = tf.Session(config=config)
```


## Optimizing the model for Serving

综合性文档： https://hackernoon.com/how-we-improved-tensorflow-serving-performance-by-over-70-f21b5dad2d98

视频课程：https://www.bilibili.com/video/av47698851


### 选择合适的指令集优化选项
编译Sering程序的时候加入优化flags，选择自己CPU所能支持的指令集。


### Batching 并发预测同一个请求中的多条样本

官方文档： https://github.com/tensorflow/serving/tree/master/tensorflow_serving/batching

问题：同一个batch的多条样本中，如果有些列特征总是相同的，如何在特征预处理的时候进行优化？

max_batch_size { value: 128 }
  - The maximum size of any batch. This parameter governs the throughput/latency tradeoff, and also avoids having batches that are so large they exceed some resource constraint (e.g. GPU memory to hold a batch's data).

batch_timeout_micros { value: 0 }
  - The maximum amount of time to wait before executing a batch (even if it hasn't reached max_batch_size). Used to rein in tail latency.

num_batch_threads { value: 8 }
  - The degree of parallelism, i.e. the maximum number of batches processed concurrently.

max_enqueued_batches { value: 1000000 }
  - The number of batches worth of tasks that can be enqueued to the scheduler. Used to bound queueing delay, by turning away requests that would take a long time to get to, rather than building up a large backlog.

- Parallelize Data Transformation

- Parallelize Data Extraction


### Batching 并发预测不同请求的样本
inter-request batching support


### “freeze the weights” of the model
tf.graph_util.convert_variables_to_constants函数

```python
from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(
    input_saved_model_dir=saved_model_dir,
    output_graph=out_graph_filename,
    saved_model_tags=tag_constants.SERVING,
    output_node_names=outpput_node_names,
    ...
)
```

### ParseExample 


### Custom DataSet OP 多线程数据预处理

### 并发处理多个请求

### GPU预测

### GTT - Graph Transform Tool

官方文档： https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms

```python
from tensorflow.tools.graph_transforms import TransformGraph

TRANSFORMS = [
  'remove_nodes(op=Identity)',
  'fold_constants(ignore_error=true)',
  'merge_duplicate_nodes',
  'strip_unused_nodes',
  'fold_batch_norms',
]

optimized_graph_def = TransformGraph(
  graph_def,
  input_names,
  output_names,
  TRANSFORMS
)
```

### 并行参数

有两个运行时的参数用于 Session parallelism。默认这两项配置是自动选择的。

intra_op_parallelism_threads 一个OP的并行, 每一个OP节点有多少个线程来支持并行

  - controls maximum number of threads to be used for parallel execution of a single operation.
  - used to parallelize operations that have sub-operations that are inherently independent by nature.

inter_op_parallelism_threads  相互独立的不同OP的并行， 最多有多少个线程来支持(不同op的节点)

  - controls maximum number of threads to be used for parallel execution of independent different operations.
  - operations on Tensorflow Graph that are independent from each other and thus can be run on different threads.

```python
  config = tf.ConfigProto()
  config.intra_op_parallelism_threads = 44
  config.inter_op_parallelism_threads = 44
  tf.Session(config=config)
```


### Client端瘦身

在Serving的时候加载 tensorflow_serving and tensorflow libraries 这两行个库增加了不必要的延时。



### SavedModel Warmup

TensorFlow Runtime 内部组件的对象策略是懒初始(Lazy Initialization)，很多对象实在真正需要的时候才会构建。也就是在第一个请求时构建。这对TF Serving也就意味着第一个请求延时会很高。

为此官方提供了warmup的解决办法，
- Warmup file name: 'tf_serving_warmup_requests'
- File location: assets.extra/
- File format: TFRecord with each record as a PredictionLog.
- Number of warmup records <= 1000.
- The warmup data must be representative of the inference requests used at serving.

参考：https://www.tensorflow.org/tfx/serving/saved_model_warmup

本质上warmup文件是tfrecord格式文件。
 

### Quantization
主要思想是通过缩小模型系数的元类型来加速和缩小内存。

32bit-float to 8bit-int 

1. Post Training Quantization
使用 TensorFlow Lite 转换器将已训练的浮点 TensorFlow 模型转换为 TensorFlow Lite 格式后，可以对该模型进行量化.

```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```


2. During Traing quantization (Quantization Aware Training)

```python
import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()


train_images_subset = train_images[0:1000] # out of 60000
train_labels_subset = train_labels[0:1000]

q_aware_model.fit(train_images_subset, train_labels_subset,
                  batch_size=500, epochs=1, validation_split=0.1)

```


### Connection Pruning







# 分布式TensorFlow (Distributed TensorFlow)

“分布式”的阶段：有训练时的分布式，有预测时的分布式（Distributed Serving）。目前TF只实现了前者。

“分布式”的内容：有模型分布式并行，有数据分布式并行。TF中一般采用数据并行， 即在各个worker节点用相同的数据流图计算不同的数据。

“分布式”的形式：有多机器的分布式，也有单机多卡的分布式。

## 重要概念
- TensorFlow server - tf.train.Server instance
	
		Master service

    Master implements the service MasterService

    Prunes a specific subgraph from the graph, as defined by the arguments to Session.run().
    Partitions the subgraph into multiple pieces that run in different processes and devices.
    Distributes the graph pieces to worker services.
    Initiates graph piece execution by worker services.
	
		Worker service

    Schedule the execution of graph operations using kernel implementations appropriate to the available hardware (CPUs, GPUs, etc).
    Send and receive operation results to and from other worker services.
	
-	Client - 在单例环境中一个graph位于一个tensorflow::Session中。对于分布式环境中，Session位于一个Server中。
	
-	Cluster - tf.train.ClusterSpec object 用于在创建 tf.train.Server 时指明spec。
	
-	Job - 一个Cluster可能包含多个Job。
	
-	Task - 一个Job可能有多个Task。

tf.train.Server.create_local_server 单进程集群，这主要是其演示作用吧。

tf.train.ClusterSpec  创建cluster配置描述

tf.train.Server 创建server实例

## TF_CONFIG

TF_CONFIG环境变量是声明cluster的标准方式。它分为cluster 和 task 两个部分。
其中cluster信息是需要在每一个节点都填写一致；task信息是每个节点填写自己的所属。
```python
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"], 'ps' : ["localhost:34567"]
    },
    'task': {'type': 'worker', 'index': 0}
})
```

type的类型有：['ps', 'worker','evaluator','chief'], There should be no "ps" job except when using tf.distribute.experimental.ParameterServerStrategy.

TF_CONFIG介绍 - https://cloud.google.com/ai-platform/training/docs/distributed-training-details

chief 是一个特殊的worker。 需要负责初始化整个运行图，其他worker 节点将从chief 节点获取计算图的信息.
主节点负责初始化参数、模型保存、概要保存.
通过调用 tf.train.MonitoredTrainingSession 来进行。

## Trainning

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/distribute/estimator_training.py


## tf.device

在模型中指明在特定节点或设备进行某个操作，比如 with tf.device('/device:CPU:0'):
```python
		with tf.device("/job:ps/task:1"):
		  weights_2 = tf.Variable(...)
		  biases_2 = tf.Variable(...)
```

## RPC Worker调用
```shell
/tensorflow.WorkerService/GetStatus
/tensorflow.WorkerService/CreateWorkerSession
/tensorflow.WorkerService/DeleteWorkerSession
/tensorflow.WorkerService/RegisterGraph
/tensorflow.WorkerService/DeregisterGraph
/tensorflow.WorkerService/RunGraph
/tensorflow.WorkerService/CleanupGraph
/tensorflow.WorkerService/CleanupAll
/tensorflow.WorkerService/RecvTensor
/tensorflow.WorkerService/RecvBuf
/tensorflow.WorkerService/Logging
/tensorflow.WorkerService/Tracing
/tensorflow.WorkerService/CompleteGroup
/tensorflow.WorkerService/CompleteInstance
/tensorflow.WorkerService/GetStepSequence
/tensorflow.WorkerService/MarkRecvFinished
```

## Replicated training 数据并行

TF中的数据并行训练又叫做 复制训练。

前面提到，TF使用的是数据并行(data parallelism)，即使不同的训练节点使用不同的数据训练完整的模型。这里的关键是如何训练出一个模型来（而不是各自训各自的），模型参数需要借助PS进行拷贝。

参考: https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md

### In-graph replication

只有一个client创建模型图和模型参数，指定把参数放在 /job:ps 上。

### Between-graph replication

每个worker都创建一个client,各个client构建相同的Graph，但是参数还是放置在ps上。TensorFlow提供了一个专门的函数tf.train.replica_device_setter来方便Graph构建.

这两种方式中更常用的是Between-graph方式.

使用tf.train.replica_device_setter可以自动把Graph中的Variables放到ps上，而同时将Graph的计算部分放置在当前worker上，省去了很多麻烦。由于ps往往不止一个，这个函数在为各个Variable分配ps时默认采用简单的round-robin方式，就是按次序将参数挨个放到各个ps上，但这个方式可能不能使ps负载均衡，如果需要更加合理，可以采用tf.contrib.training.GreedyLoadBalancingStrategy策略。
`By default, only Variable ops are placed on ps tasks, and the placement strategy is round-robin over all ps tasks. A custom ps_strategy may be used to do more intelligent placement, such as tf.contrib.training.GreedyLoadBalancingStrategy.`

[How does "tf.train.replica_device_setter" work?](https://stackoverflow.com/questions/39666845/how-does-tf-train-replica-device-setter-work)

[数据并行时两种图复制模式的比较](https://stackoverflow.com/questions/41600321/distributed-tensorflow-the-difference-between-in-graph-replication-and-between)

## PS in Tensorflow

参数服务器(parameter server)，简称为ps，用于存储可训练的参数变量tf.Variable.

ps作为tensorflow分布式训练中作为一个worker。

## 同步模式计算更新梯度

tf.train.SyncReplicasOptimizer 


## 异步模式计算更新梯度



## 数据并行下的分布式数据输入

https://www.tensorflow.org/tutorials/distribute/input

数据并行时，每个节点的训练数据都不同。




















# Tensorflow 训练示例

例子教程:
https://github.com/aymericdamien/TensorFlow-Examples
https://github.com/nlintz/TensorFlow-Tutorials
https://codelabs.developers.google.com/?cat=TensorFlow
https://github.com/tensorflow/models

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

这是一个服饰类的图像数据集，包含了10个类别，分别是10种服饰类型。一共7万张图片。

keras.datasets.fashion_mnist.load_data()


## ImageNet 图像数据集模型训练

ImageNet的图像分类数据集叫做ILSVRC。ILSVRC图像分类数据集包含了来自1000个类别的120万张图片，其中每张图片属于且只属于一个类别。

### GoogLeNet

https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py

### AlexNet

https://github.com/tensorflow/models/blob/master/tutorials/image/alexnet/alexnet_benchmark.py


### LFW

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


