---
title: "TensorFlow实用手册"
categories: [MachineLearning]
layout: post
---

本篇所涉及的TensorFlow API都在官方文档有所涉及，https://www.tensorflow.org/api_docs/

首先要注意，tensorflow版本之间差异比较大，一些API会发生增减或者位置迁移。

# 框架体系

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

最适合用于模型实践的API就是tf.estimator这一套方法。

### tf.estimator.Estimator 类

tf.estimator是一个基类。

可以使用原生预设的模型子类，比如 DNNClassifier、 DNNRegressor等

也可以基于基类自己实现子类。

```
predictor = tf.estimator.Estimator(
        model_fn=model.model_fn,
        params={  # dict of hyper parameters that will be passed into model_fn
            'feature_columns': columns,
            'config': config,
            'args': args,
        },
        model_dir=config.model_path,
        log_dir =config.tensorboard_dir,
        config=run_config,  # tf.estimator.RunConfig
        warm_start_from=ws  # tf.estimator.WarmStartSettings
      )
```
其中的 params["feature_columns"]是在模型中的所有FeatureColumns组成的列表或字典。后面会有一章单独讲Feature Columns。

### model_fn Spec

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
eval_metric_ops由若干tf.metrics指标模块所组成的字典，比如tf.metrics.accuracy,tf.metrics.precision,tf.metrics.recall,tf.metrics.auc.
3. 如果是predict任务，需要输入predictions 来构建 EstimatorSpec


### tf.estimator.train_and_evaluate 函数

这个方法是真正去训练模型。它的输入是 Estimator对象 + TrainSpec对象 + EvalSpec对象。

```
# 上一节代码里创建有 estimator
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

### input_fn Spec 

我们需要实现 input_fn 方法来创建TrainSpec和EvalSpec并返回，用于tf.estimator.train_and_evaluate.

tf.estimator.TrainSpec  用来定义输入的训练数据，需要传入 input_fn=train_input_fn.

tf.estimator.EvalSpec  用来定义eval部分的配置，需要传入 input_fn=eval_input_fn.

input_fn 作为TrainSpec/EvalSpec最重要的输入参数，它是一个方法，该方法最终应该返回是数据。可以支持的类型有两种：

 1. A tuple (features, labels):

 2. A 'tf.data.Dataset' object: 该Dataset的返回值要是 tuple (features, labels) 


### Config

tf.estimator.RunConfig 各种配置都填在这个类对象中。它会被作为conf参数用于构建 Estimator。

tf.estimator.ModeKeys  设定当前的工作模式（eval、predict、train）

tf.estimator.WarmStartSettings 它被作为warm_start_from参数用于构建 Estimator。

tf.estimator.VocabInfo  表示 WarmStartSettings 的词汇信息。它被用于构建WarmStartSettings.



## Tensorflow Eager 模式 API

无需构建图：操作会返回具体的值，而不是构建以后再运行的计算图

https://www.tensorflow.org/guide/eager






## 可视化模块 Tensorboard

官方文档：https://github.com/tensorflow/tensorboard/blob/master/README.md

### tensorboard 命令

		tensorboard --logdir=/path/to/log-directory

tensorboard默认占用了6006端口

		lsof -i:6006

如果想查找当前目录里面有多少文件可以被可视化出来，可以用inspect参数来扫描目录。路径可以是相对路径。

		tensorboard --inspect  --logdir=./

### tf.summary API

tf.summary 提供了向文件写入模型内部的结构和数据信息的方法，以供 tensorboard 来展示。

tf.summary.merge_all()  可以将所有summary全部保存到磁盘，以便tensorboard显示.

tf.summary.FileWriter('xxx', sess.graph)

https://www.tensorflow.org/api_guides/python/summary

一个例子：https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_basic.py


### 可视化中间文件 tfevent

events.out.tfevents.XXX.local 文件是summary方法所生成的文件，其中包含了用于tensorboard进行可视化展示所需的信息。
每创建一个tf.summary.FileWriter实例，就会对应的生成一个tfevent文件。

Event files, which contain information that TensorBoard uses to create visualizations.

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

tensorflow::GraphDef是图，模型的载体。
tensorflow::Session是训练或预测过程的会话载体。
tensorflow::Tensor是模型输入、输出的载体。

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
```
def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
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

### tf.get_variable

tf.Variable与tf.get_variable()的区别是：
tf.get_variable() 会检查当前命名空间下是否存在同样name的变量，可以方便共享变量。而 tf.Variable 每次都会新建一个变量。
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


## tf.placeholder 可变数据输入

在构建模型的时候没必要先把数据写好，先用tf.placeholder把数据类型确定就行。在真正session执行的时候再用feed_dict把输入填进去就好。


## tf.Operation

运算节点。Operation节点的输入是tensor或0，输出是tensor或0.

在graph.pbtxt文件中能看到每一个node里，都有一个key名为op的字段，它指明了对tensor对象的操作。



The `local_init_op` is an `Operation` that is run always after a new session was created.

### get_operation_by_name 从graph中由名字获取到op
graph.get_operation_by_name(op_name)


## custom_ops

custom op指的是使用C++来实现自己的tensor操作。

https://www.tensorflow.org/guide/extend/op

https://github.com/tensorflow/custom-op

### 定义自定义op的接口
```
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
using namespace tensorflow;
REGISTER_OP("接口名称")
    .Input("输入名称: int32")
    .Output("输出名称: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("接口名称").Device(DEVICE_CPU), 类名);
```
- 其中input和output都是在定义输入输出tensor的名称和类型
- 其中还实现了一个Shape functions: infers the shape of outputs given that of inputs.

### 编写自定义op的内部实现

After you define the interface, provide one or more implementations of the op. To create one of these kernels, create a class that extends OpKernel and overrides the Compute method. The Compute method provides one context argument of type OpKernelContext*, from which you can access useful things like the input and output tensors.

需要实现一个compute方法

```
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
```
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

### 使用自定义op

```
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

### Dataset ops

实现一个 from tensorflow.python.data.ops import dataset_ops 的子类，然后将该类对象传入到 input_fn .



## tf.Session 运行数据流

在 tf.Session 之前的过程都是定义，tf.Session().run(...)才是真正执行前面定义好的操作。如果只声明tensor而运行session.run，是不会运行计算逻辑的。

Run函数 是整个tensorflow graph的核心过程。

首先看 run函数的接口

		run(
		    fetches,
		    feed_dict=None,
		    options=None,
		    run_metadata=None
		)

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

```
import tensorflow as tf
#导入命令行解析模块
import argparse
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





# 常用函数

用以下API完成日常工作。 包括基础操作方法、模型保存加载方法、模型流图构建方法、模型训练方法。

## 基础操作函数 Common Function

先看一些基础的操作函数。

### 基本运算函数

- tensor向量随机生成
```
tf.random_normal

tf.random_uniform

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
```
tf.concat
在某个维度把两个tensor串联起来。

tf.sparse_concat
在某个维度把两个 sparse_concat 串联起来。
```

- 关于tensor向量的判断
```
tf.equal

tf.where
tf.where(condition, x = None, y = None, name = None)，根据condition判定返回。即condition是True，选择x；condition是False，选择y。
```

### 向量标准化

tf.nn.l2_normalize

### 类型形式转换函数

https://www.tensorflow.org/api_guides/python/array_ops

```
tf.cast

tf.expand_dims 增加一个维度，被增加的维度的数据长度就是1.

tf.reshape 

tf.squeeze 将原始input中所有维度为1的那些维都删掉

tf.tile 对当前张量内的数据进行一定规则的复制。最终的输出张量维度不变。
```

### tensorflow::Flag

用于解析和处理命令行参数


## 模型保存和加载函数

我们经常在训练完一个模型之后希望保存训练的结果，这些结果指的是模型的参数，以便下次迭代的训练或者用作测试。

1. 第一种：是传统的 tf.train.Saver 类save保存和restore恢复方法。Tensorflow针对这一需求提供了Saver类。
这种方法将模型保存为ckpt格式。

tf.train.get_checkpoint_state   输入路径必须是绝对路径

```
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

```
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

```
export_savedmodel(
    export_dir_base,
    serving_input_receiver_fn,
    assets_extra=None,
    as_text=False,
    checkpoint_path=None,
    strip_default_attrs=False
)
```


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

- 交叉熵损失函数 softmax_cross_entropy_with_logits

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

优化器有哪些？
```
tf.train.AdamOptimizer
tf.train.GradientDescentOptimizer
```

优化器怎么用？
```
my_opt = tf.train.GradientDescentOptimizer(0.02) # 参数时学习率
train_step = my_opt.minimize(loss) # 其中的loss是自己经过网络之后又构建好的损失值tensor
```

优化器函数是怎么更新整个网络参数的？
通过operation。 my_opt.minimize(loss)得到的就是一个op，把这个op传入到session.run(train_step)里面去，就会更新网络的权值。

```
train_op = optim.minimize(loss, global_step=self.global_step, var_list=train_vars)
```
var_list 参数指明了本次优化中可以被更新的权值。
global_step 参数是训练迭代的计数器，比如说在Tensorboard画loss和 accuracy的横坐标即是global_step。优化器op每执行一次，该值就会自增1.







# 特征处理 Feature Columns

Feature Columns是Tensorflow中 原始数据 和 Estimator 的中间转换，这一过程是把换数据转换为适合Estimators使用的形式。机器学习模型用数值表示所有特征，而原始数据有数值型、类别型等各种表示形式。Feature Columns其实就是在做特征预处理。

## 如何使用 Feature Columns？

feature_columns 作为 `Estimators的params参数`之一，它将输入数据 input_fn 和 模型 联系起来。
我们输入到`input_fn`中的训练数据也是依据feature_columns的格式生成的。可以看到 tf.feature_column 有很多种。

可参考 https://www.tensorflow.org/guide/feature_columns

- tf.feature_column.input_layer() 比较特殊，它作为输入层。

- tf.feature_column.make_parse_example_spec 方法将若干个feature_colunms转换为key-value字典形式（key是feature name， value是FixedLenFeature 或 VarLenFeature）

```
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

## Embedding column
tf.feature_column.embedding_column
对类型特征进行Embedding编码后的特征。

以上这两种colunm是以Categorical column为输入基础的，


## tf.feature_column.shared_embedding_columns
若干个embedding column共享一模一样的权重数值。

## tf.feature_column.weighted_categorical_column
Applies weight values to a CategoricalColumn







# 样本文件/数据格式处理

tf.Example messages to and from tfrecord files

## 数据输入流
样本输入的过程称作是ETL过程，这一过程由Extract、Transform、Load三个步骤组成。
- Extract是从硬盘or网络磁盘到内存的过程。
- Transform是在内存中进行格式转换，比如从 protobuf 到 tf.data.Dataset.
- Load是将batch规模的样本加载到GPU加速设备上.


## tf.Example / tf.train.example

TFRecord是文件形态，tf.train.Example就是内存对象形态.

tf.Example is a {"string": tf.train.Feature} mapping.

tensorflow的 example 包含的是基于key-value对的存储方法，其中key是一个字符串，其映射到的是feature信息，feature包含三种类型：
		BytesList：字符串列表
		FloatList：浮点数列表
		Int64List：64位整数列表


### tf.train.SequenceExample

### tf.parse_example

parse_example 方法把序列化的特征解析为字典类型(tensor)。
参考 tensorflow/python/ops/parsing_ops.py.
parse_example的输入：
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos.
    features: A mapping dict from keys to `VarLenFeature`, `SparseFeature`, and `FixedLenFeature` objects.

VarlenFeature： 是按照键值把example的value映射到SpareTensor对象.
FixedLenFeature：是按照键值对将features映射到大小为[serilized.size(), df.shape]的Tensor矩阵.
SparseFeature：稀疏表示方式的feature，不推荐使用。

parse_example的输出：
    return: A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.


tf.train.Feature

### tf.parse_single_example

区别于tf.parse_example，tf.parse_single_example只是少了一个batch而已，其余的都是一样的


## TFRecord

TFRecord是Tensorflow特有的二进制数据存储格式。它的好处是性能，在加载和传输时代价较小。另一个好处是可以存储序列化数据。

我们用Tensorflow API可以方便的构建和读写TFRecord数据。

tf.python_io.TFRecordWriter

使用tf_record_iterator方法可以从tfrecord文件中解析出json(k-v)形式的特征数据。
```
import tensorflow as tf
target_file = "tf_record_file_000"
for example in tf.python_io.tf_record_iterator(target_file):
    result = tf.train.Example.FromString(example)
```


## tf.data.Dataset

tf.data.Dataset协助我们完成数据从文件形式到灌入Tensor的处理过程。
在训练模型的时候，tf.data.Dataset 可以作为 input_fn 方法的返回值数据.
在进行预测的时候，tf.data.Dataset 

下面的七行代码，我们使用tf.data.Dataset来完成ETL三个过程。
```
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


## 自定义文件格式

设计自定义文件格式和自己的方法构建tensor，需要自己实现两个任务：
1. 文件格式：使用 tf.data.Dataset 阅读器来从文件中读取原始记录（通常以零阶字符串张量（scalar string tensors）表示，也可能有其他结构）。
2. 记录格式：使用解码器或者解析操作将一个字符串记录转换成 TensorFlow 可用的张量（tensor）。

- DatasetOpKernel 的子类

要自己实现一个 tensorflow::DatasetOpKernel 的子类，这个类的 MakeDataset() 方法要告诉 TensorFlow 怎样根据一个操作的输入和属性生成一个数据集的对象。

- MakeDataset 方法要返回一个 DatasetBase 的子类

要自己实现 DataSetBase 的子类，这个类的 MakeIteratorInternal() 方法 要构建迭代器。

- DatasetIterator 的子类

一个 tensorflow::DatasetIterator<Dataset> 的子类，表示特定数据集上的迭代器的可变性，这个类的 GetNextInternal() 方法告诉 TensorFlow 怎样获取迭代器的下一个元素。

GetNextInternal 定义了怎样从文件中实际读取记录，并用一个或多个 Tensor 对象来表示它们.

GetNextInternal 可能会被并发调用，所以推荐用一个互斥量来保护迭代器的状态。

    EnsureRunnerThreadStarted

      RunnerThread  通过StartThread开启的线程函数
        CallFunction
          map_func

      ProcessResult

    CallCompleted 释放锁







# 模型文件格式

1. GraphDef
2. SavedModels


下面两种模型文件格式对应着tensorflow的两种模型文件保存方式。

checkpoint文件 是用于本地加载模型然后进行本地预测的。
pb-variable文件是用来让tensorflow serving加载并进行远程预测的。

在模型文件中，我们想保存的信息有两种：
1. a graph (various operations).
2. weights/variables in a graph.

谷歌推荐的保存模型的方式是保存模型为 PB 文件，它具有语言独立性，可独立运行，封闭的序列化格式，任何语言都可以解析它，它允许其他语言和深度学习框架读取、继续训练和迁移 TensorFlow 的模型。

## checkpoint文件

这是由 tf.train.Saver 类生成的模型文件。

The .ckpt is the model given by tensorflow which includes all the 
weights/parameters in the model.

check_point文件，包含三个主要文件，meta, index, data。
meta主要有各种def，一个很重要的就是graph_def，而data保存真正的weight。

checkpoints, which are versions of the model created during training. 存储的为最近的几次迭代保存的模型名称以及路径：

		meta file: 在meta文件中保存的是模型的图。describes the saved graph structure, includes GraphDef, SaverDef, and so on; then apply tf.train.import_meta_graph('/tmp/model.ckpt.meta'), will restore Saver and Graph.
	
		index file: 在index文件中保存的为模型参数的名称以及具体属性。it is a string-string immutable table(tensorflow::table::Table). Each key is a name of a tensor and its value is a serialized BundleEntryProto. Each BundleEntryProto describes the metadata of a tensor: which of the "data" files contains the content of a tensor, the offset into that file, checksum, some auxiliary data, etc.
	
		data file: 在data文件中保存的为模型参数的数值。it is TensorBundle collection, save the values of all variables.

https://www.tensorflow.org/guide/checkpoints


## pb-variable文件

这是由 tf.saved_model.builder.SavedModelBuilder 类生成的模型文件。

variables保存所有变量; saved_model.pb用于保存模型结构等信息。

pb文件，其实就是graph_def，但是指的一般是做了constant化，这样可以直接加载做inference，安装部署用。
The .pb file stores the computational graph. Includes the graph definitions as `MetaGraphDef` protocol buffers.
PB是表示 MetaGraph 的 protocol buffer格式的文件，MetaGraph 包括计算图，数据流，以及相关的变量和输入输出signature以及 asserts 指创建计算图时额外的文件。

assets目录文件：assets is a subfolder containing auxiliary (external) files, such as vocabularies. Assets are copied to the SavedModel location and can be read when loading a specific MetaGraphDef.

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

### pb 和 pbtxt

There are actually two different formats that a ProtoBuf can be saved in. 

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
```
tf.saved_model.builder.SavedModelBuilder

tf.saved_model.utils.build_tensor_info

tf.saved_model.signature_def_utils.build_signature_def

builder.add_meta_graph_and_variables

builder.save()
```


## checkpoint文件 和 pb-variable文件之间的转换






# 模型训练方式

## Multi-head / Multi-task DNN

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


```
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







# Transfer Learning - Retrained Model

通过迁移学习，我们不需要太多的数据！这个想法是从一个以前在数百万图像上训练过的网络开始的，比如在ImageNet上预训练的ResNet。然后，我们将通过仅重新训练最后几个层并使其他层独立来微调ResNet模型。

通常我们会把已有模型中前面的几层直接拿过来用，重新设计后面几层的结构，重新训练后面几层的权值，注意新训练的时候要把前面几层的权值固定。

bottleneck指的是网络最后输出层之前的一层（倒数第二层）。这一层中原始的特征已经经过了前面若干层而被压缩到了新的表示空间。对于图像分类网络来讲，它就是image feature vector。

## bottleneck layer - penultimate layer


## final layer retraining


## Fine Tune
Fine Tune通常指的就是冻结网络前面的层，然后训练最后一层。

在调用优化器的 minimize 方法生成训练op的时候，可以传入一个参数var_list来指定可以被优化的参数。
```
output_vars = tf.get_collection(tf.GraphKyes.TRAINABLE_VARIABLES, scope='outpt')
train_step = optimizer.minimize(loss_score,var_list = output_vars)
```


## retrain一个model

```
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



## 使用retrained model进行预测

```
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

如果编译环境的内存不够大或者gcc版本过高，编译时就容易遇到编译系统错误。所以我编译的时候主动用的较低的gcc版本4.9来编译。

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
	|       |-- saved_model.pb   保存了serialized tensorflow::SavedModel.
	|       `-- variables   保存了variables.
	|           |-- variables.data-00000-of-00001
	|           `-- variables.index
```

3. 然后将这个模型载入到 TensorFlow ModelServer，注意输入的模型路径必须是绝对路径。

    tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/

或者使用配置文件加载模型

    tensorflow_model_server --port=9000 --model_config_file=/serving/models.conf


4. gRPC方式对外提供服务

默认使用 --port方式就是以gPRC方式提供服务。

```
$ tensorflow_model_server \
      --port=9000 \
      --model_name=mnist \
      --model_base_path=/tmp/mnist_model/
```

5. RESTful方式对外提供服务

用参数指明要使用rest方式提供服务。
```
$ tensorflow_model_server \
   --rest_api_port=8501 \
   --model_name=half_plus_three \
   --model_base_path=$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_three/
```

## 客户端 tensorflow-serving-api

在客户端把样本数据作为请求发送到TensorFlow ModelServer，

		python tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=localhost:9000

## Client-Server 交互过程

具体的交互流程一般是这样：

```
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

```
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

Feature.proto 和 example.proto
定义在tensorflow/core/example/feature.proto和tensorflow/core/example/example.proto


TensorProto
TensorProto是一个pb message，定义在tensorflow/core/framework/tensor.proto，用来表示一个Tensor。

TensorInfo


SignatureDef
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

SignatureDefMap
由 name->SignatureDef 构成的map。

MetaGraphDef
由一个计算图 GraphDef 和其相关的元数据（SignatureDef、CollectionDef、SaverDef）构成。其包含了用于继续训练，实施评估和（在已训练好的的图上）做前向推断的信息。
定义在tensorflow/core/framework/graph.proto


PredictRequest
由 map<string, TensorProto> 作为请求输入。要预测的样本就放在其中。

PredictResponse
由 map<string, TensorProto> 作为请求返回。预测的结果就放在其中。


## input receiver 解析输入

对于Serving来说，预测时的输入data就是 `tf.example` 形式的.

serving_input_receiver_fn 方法在serving阶段，相当于训练阶段的 input_fn 方法。
它返回了一个 ServingInputReceiver 对象。 这个对象创建时传入了两个参数：
  一个是 features=parsing_ops.parse_example 的返回值，它定义了传给模型的features.
  一个是 receiver_tensors={receiver_key: serialized_tf_example}.
它是要把 tf.Example 解析为 tensor. 

serving_input_receiver_fn 是在导出模型时被使用的。
在导出模型的时候，会将 serving_input_receiver_fn 方法传入到 export_savedmodel 方法中。


tf.estimator.export.ServingInputReceiver 和 tf.estimator.export.TensorServingInputReceiver 有一点点差异：
`tf.estimator.export.TensorServingInputReceiver` allows `tf.estimator.Estimator.export_savedmodel` to pass raw tensors to model functions.


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






## Optimizing the model for Serving

0. Batching 并发预测同一个请求中的多条样本
max_batch_size { value: 128 }
batch_timeout_micros { value: 0 }
max_enqueued_batches { value: 1000000 }
num_batch_threads { value: 8 }

Parallelize Data Transformation

Parallelize Data Extraction


0. Batching 并发预测不同请求的样本
inter-request batching support


1. “freeze the weights” of the model

tf.graph_util.convert_variables_to_constants函数

2. Custom DataSet OP 多线程数据预处理

3. 并发处理多个请求

4. GPU预测






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







# Tensorflow 机器学习模型

https://github.com/aymericdamien/TensorFlow-Examples/tree/master/examples/2_BasicModels

	SVM
	kmeans
	线性回归
	逻辑回归
	KNN
	随机森林

例子教程:
https://github.com/aymericdamien/TensorFlow-Examples
https://github.com/nlintz/TensorFlow-Tutorials
https://codelabs.developers.google.com/?cat=TensorFlow
https://github.com/tensorflow/models



## 神经网络模型

tf.nn.bidirectional_dynamic_rnn
dynamic version of bidirectional recurrent neural network. 


tf.nn.rnn_cell.GRUCell
Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).














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
