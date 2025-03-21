---
layout: post
title:  "ONNXRuntime推理引擎"
subtitle: "ONNXRuntime"
categories: [MachineLearning]
---

# 图相关的概念

GraphViewer 是对计算图的表示，包含节点（Node）和边（Edge），节点代表计算操作，边代表数据流。

FusedNodeAndGraph

在模型交给provider执行之前，对图做了些什么？


## TransformGraph
```该步骤发生在plan之前，在这里确定什么节点在什么设备上执行```
1. 设备分配
2. 图优化
3. 插入跨设备之间需要的copy节点


# 执行机制、设备分配、内存管理

## stream的概念

cpu stream意味着是什么？

gpu stream意味着什么

plan执行计划中，有一个成员```std::vector<InlinedVector<NodeIndex>> stream_nodes_;```


## plan的概念
```plan > stream > step```
SequentialExecutionPlan 的组成部分
* execution_plan：
    * execution_plan 由vector of LogicStream 组成
        * LogicStream 由vector of ExecutionStep 组成
* allocation_plan：
    * tensor边的内存分配信息


## plan的构建
```CreatePlan :  PartitionIntoStreams -> BuildExecutionPlan```
1. 决策要分多少个stream, stream的个数默认遵循ep/device个数
2. 创建stream实例
3. 在stream中创建step

## plan的执行

* 不设置 ORT_ENABLE_STREAM -> 全局只有一个流 -> 同步执行模式 
* 基于device分配


stream 和 并行 的关系？

stream 和 执行方式 的关系？

### 一次 LaunchKernelStep 意味着什么？
LaunchKernelStep 是真正执行算子的step，其他的都是控制执行时序的(cross-stream synchronization)。
* LaunchKernelStep 
* BarrierStep
* TriggerDownstreamStep
* WaitOnEPStep
* ActivateNotificationStep

## ExecutionMode::ORT_PARALLEL 意味着什么？



# provider的概念



# onxxruntime 线程模型
ONNXRuntime的线程池接口在Eigen线程池接口基础之上扩展而来（题外话：TensorFlow中的线程池同样是建立在Eigen线程池基础上）

线程池维护一组系统线程（OS threads）,用于执行ThreadPoolTempl::WorkerLoop。每个线程都拥有自己的运行队列（RunQueue），运行队列中都是被push进来的待执行的任务（Task）。主要的工作任务是从队列中弹出一个任务并执行至结束。如果线程的运行队列为空，则线程陷入自旋（spin）等待任务到达，并且尝试从其它线程的运行队列中“偷取”任务来执行，如果没有偷来任务，则阻塞在系统中。在创建线程池时会通过配置标志（flag）和常量 spin_count 来实现这种“spin-then-block”操作；

* inter_op_num_threads 不同算子的并行，使用eigen线程池
* intra_op_num_threads 同一个算子的并行，使用openmp实现


use_per_session_threads ：是否为每个会话使用单独的线程池。 默认是否

intra_op_thread_pool_from_env_ = session_env.GetIntraOpThreadPool();
inter_op_thread_pool_from_env_ = session_env.GetInterOpThreadPool();

allow_spinning


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



# 了解一些算子

* shape 很好理解，就是算出输入tensor的维度，然后作为一个一维tensor输出出来  
* reshape 输入数据tensor和期望的维度tensor，输出调整维度后的tensor （如果转换不了，会执行失败）
* tile 输入数据tensor和repeats tensor，其中repeats tensor是一维tensor，每一个元素意味着对应data维度要重复复制多少遍，输出扩张复制后的新tensor
* squeeze 移除张量中维度大小为1的指定轴
* Unsqueeze 在指定位置插入新的轴，从而增加张量的维度
* slice 在axes指定的维度上，从starts到ends做切片