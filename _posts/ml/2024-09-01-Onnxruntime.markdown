---
layout: post
title:  "ONNXRuntime生态"
subtitle: "ONNXRuntime"
categories: [MachineLearning]
---


# 图相关的概念

FusedNodeAndGraph

GraphViewer

在模型交给provider执行之前，对图做了些什么？

# 执行机制、设备分配、内存管理


# provider




# 注册算子的几种方法


* ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME
    * 入参： provider名、onnxdomain、最低版本、最高版本、数据类型、算子名称
    * 有选择的支持部分数据类型
* ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME
    * 入参：provider名、onnxdomain、最低版本、数据类型、算子名称
* ONNX_OPERATOR_KERNEL_CLASS_NAME
    * 入参： provider名、onnxdomain、最低版本、算子名称
    * 意思是任何数据类型都能支持
* ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME
    * 入参： provider名、onnxdomain、最低版本、最高版本、算子名称


* ONNX_OPERATOR_VERSIONED_KERNEL_EX
    * 入参： 算子名、onnxdomain、最低版本、最高版本、provider名
* ONNX_OPERATOR_TYPED_KERNEL_EX

* ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX


ONNX_OPERATOR_KERNEL_EX



# 算子按开发模式分类

* onnxruntime/onnxruntime/core/providers 
    * https://github.com/onnx/onnx/blob/main/docs/Operators.md
* onnxruntime/onnxruntime/contrib_ops
    * https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
* Custom operators 代码不编入onnxruntime库内部




# Python运行

```python
import onnxruntime as ort

# Load custom op library
so = ort.SessionOptions()
so.register_custom_ops_library('my_custom_op.so')

# Create session
session = ort.InferenceSession('model.onnx', so)
```


# C++运行







# onnxruntime代码库内自带的算子融合

## 看看 EmbedLayerNormalization 是如何融合成的？

enable_embed_layer_norm -> fuse_embed_layer -> FusionEmbedLayerNormalization

```
BERT Embedding Layer Pattern:
                            (input_ids)
                           /         \
                          /          Shape
                         /              |
                        /              Gather (indices=1)
                       /                  |
                      /                 Add (optional, B=0)
                     /                    |
                Gather   segment_ids  Unsqueeze (axes=0)
                    \        |           |
                     \     Gather      Slice (data[1,512], starts=0, ends=*, axes=1, steps=1)
                      \    /            |
                        Add          Gather
                         \        /
                             Add
                              |
                        LayerNormalization
```

输入：
    input_ids, 
    segment_ids (也就是 模型的输入token_type_ids),   
    word_embedding,  （内置向量词表）
    position_embedding, （内置向量词表）
    segment_embedding,  （内置向量词表）
    gamma,   （模型 Mul算子里的 LayerNorm.weight ）
    beta,   （模型 Add算子里的 LayerNorm.bias）
    mask (也就是 模型的输入att_mask), 
    position_ids （input_ids经过位置编码过程得到的）

mask是怎么使用的？     mask ->  mask_index 输出将来给attention使用

UT: embed_layer_norm_op_test.cc



## 看看 SkipLayerNormalization 是如何融合成的？

输入：
    input
    skip   （前一个SkipLayerNormalization的输出）
    gamma （也就是 attention.output.LayerNorm.weight）
    beta  （也就是 attention.output.LayerNorm.bias）
    bias  （也就是 attention.output.dense.bias）  先做Add

skip 是怎么使用的？  先把input和skip做Add

```
    input                      skip
      \                        /
     Add(dense.bias)        Add
          \                 /
        LayerNormalization(gamma, beta)
```

UT: skiplayernorm_op_test.cc




# onxxruntime 线程模型
ONNXRuntime的线程池接口在Eigen线程池接口基础之上扩展而来（题外话：TensorFlow中的线程池同样是建立在Eigen线程池基础上）

* inter_op_num_threads 不同算子的并行，使用eigen线程池
* intra_op_num_threads 同一个算子的并行，使用openmp实现




# onnxruntime中的四种量化校准方法

class CalibrationMethod(Enum):
    MinMax = 0
    Entropy = 1
    Percentile = 2
    Distribution = 3

最基础的是MinMax

* augment_graph 向模型中添加了 ReduceMin 和 ReduceMax 节点，用于计算每个张量的最小值和最大值。这些节点的输出被添加到模型的输出中。
* collect_data 从提供的数据读取器中获取输入数据，并使用增强的模型来推断这些数据。推断结果被存储在 intermediate_outputs 属性中。
* compute_data 处理收集的数据，计算每个张量的最小值和最大值。这些值可以通过简单的最小值和最大值计算或使用移动平均来平滑。

除了minmax之外的三种方法均派生自 HistogramCalibrater

* collect_data 方法从提供的数据读取器中获取输入数据，并使用增强的模型来推断这些数据。推断结果被存储在 intermediate_outputs 属性中，并且还会生成每个张量的直方图
* compute_data 方法根据收集的数据和直方图计算每个张量的量化参数。具体的计算方法取决于所选的方法（如百分位数或熵）。
    * 在百分位数方法中，会根据指定的百分位数（如 99.999%）计算张量的最大和最小值。
    * 在熵方法中，会通过最大熵的方式来确定张量的量化参数。