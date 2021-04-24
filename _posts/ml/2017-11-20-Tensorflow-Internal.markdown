---
layout: post
title:  "Tensorflow 内部设计模式"
subtitle: "Tensorflow Internal"
categories: [MachineLearning]
---

Tensorflow 作为近年来最为流向的机器学习框架，在内部实现上有非常多值得学习的地方。这篇博客不涉及Tensorflow的用法，有关Tensorflow的使用请参考另一篇《Tensorflow使用手册》。

# Tensorflow 框架体系的设计模式

[官网文档](https://www.tensorflow.org/guide/extend/architecture) 中介绍了一些TF框架的实现机制。

- 支持异构设备

- 支持异构语言

- Tensor 数据形式的统一化

- Protobuffer 结构定义的统一化

- OP 计算逻辑的统一化

https://gist.github.com/dustinvtran/cf34557fb9388da4c9442ae25c2373c9

- 前端系统和后端系统

前端系统是多语言的编程环境， 后端系统是C++实现。





# 源代码组织结构

tensorflow/core  核心代码由C++实现。

　　core/ops/ contains the "signatures" of the operations
　　core/kernels/ contains the "implementations" of the operations (including CPU and CUDA kernels)
　　core/framework/ contains the main abstract graph computation and other useful libraries
　　core/platform/ contains code that abstracts away the platform and other imported libraries (protobuf, etc)


tensorflow/contrib



## 从源码编译Tensorflow

https://tensorflow.google.cn/install/source?hl=zh-cn

### 编译python pip包
```shell
    cd tensorflow-1.15

    #配置 build
    ./configure

    #构建 pip 软件包
    bazel build --copt=-march=native --copt="-Wno-error" --config=noaws --config=nogcp --config=nohdfs --config=nokafka //tensorflow/tools/pip_package:build_pip_package

    # 构建出wheel包
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package  /tmp/tensorflow_pkg

    # 安装wheel包
    pip install /tmp/tensorflow_pkg/tensorflow-version-tags.whl
```



# C-API 跨语言支持 

大多数情况下，我们使用Python来进行模型训练，所有可用的Python API都在 https://tensorflow.google.cn/api_docs/python

也可以使用C++来进行模型训练， 所有可用的C++ API都在 https://tensorflow.google.cn/api_docs/cc

Python API具备功能最为全面的方法，能够支持基本上机器学习工作中所需要的几乎所有操作。

### Python API 和 C++ API是如何对应和调用的呢？

在tensorflow/core/kernels目录下，能看到非常多的xxx_op.cc，其实Python调用到的OP方法也都是C++实现。

Tensorflow在编译时生成gen_array_ops.py

通过注册 REGISTER_KERNEL_BUILDER

- 有些运算操作的对应关系比较直接：

比如 tf.unique 和 class UniqueOp
比如 tf.concat 和 class ConcatBaseOp
比如 tf.argmax 和 class ArgMaxOp
比如 tf.reshape 和 class ReshapeOp
比如 tf.matmul 和 class MatMulOp


- 有些运算操作是比较间接的：

比如 tf.reduce_xxx
```cpp
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Max")                                                              \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tidx"),                                      \
      ReductionOp<CPUDevice, type, int64, Eigen::internal::MaxReducer<type>>);
```


- 巧妙应用的python的装饰器，提高了代码的动态性。
```cpp
tf_export = functools.partial(api_export, api_name=TENSORFLOW_API_NAME)
estimator_export = functools.partial(api_export, api_name=ESTIMATOR_API_NAME)
keras_export = functools.partial(api_export, api_name=KERAS_API_NAME)
```

# OpKernel

tensorflow/python/ops
tensorflow/core/ops

- OP注册操作 REGISTER_OP 的实现
将op name 和 OpRegistrationData 关联起来，保存到 registry_ 这个map中。

- Kernel注册操作 REGISTER_KERNEL_BUILDER 的实现
创建一个名称唯一， 类型为 OpKernelRegistrar 的全局静态变量



## FunctionDefHelper 是怎么工作的？






# 计算图

计算图中的Node实例 就是 OP 
计算图中的Edge实例 就是 Tensor





# Session

SessionFactory






# 梯度是如何计算的？

我们要在Tensorflow Graph中计算梯度，当 Session 运行起来的时候，TF会自动按图运算流向的反方向生产一张逆向的图。

我们自己创建的图叫做 Forward Pass Graph， 它由输入数据(placeholder)、运算单元(OP)、和模型参数（Variables）构成。而TF自动生成的对应的图叫做 Backward Pass Graph。

## GradientDef

GradientDef 定义了一个函数对应的梯度函数。








# Runtime







# StreamExecutor

tensorflow/stream_executor

https://github.com/henline/streamexecutordoc

https://www.cnblogs.com/deep-learning-stacks/p/9386188.html







# Compiler, XLA 

Accelerated LinearAlgebra

XLA是TensorFlow图表的编译器，该组件的目标是加速TF数据流图的执行，提高内存效率，降低操作依赖。

XLA主要由数据流图转换器、XLA编译器、JIT（just-in-time）编译机制、AOT（ahead-of-time）编译机制等模块构成。

XLA的核心是高阶优化中间表示层（HLO IR，High Level Optimization Intermediate Presentation）。HLO是一种面向线性代数语义的编译器，Tensorflow官网或源码中的operation_semantic.md文档给出了HLO的操作说明。HLO的引入将核函数开发时的前后端代码解耦，有助于增强Tensorflow的可移植性。

XLA技术的总体流程是将Tensorflow的数据流图转换为XLA图，再由基于LLVM的编译器生成相应设备的二进制文件。

参考: https://zhuanlan.zhihu.com/p/124269986



# 幕后英雄 Thirdparty

在 third_party 下包含了tensorflow依赖的第三方库，有些是Google自己的开源项目，有的是外部的项目。

Protobuffer - 数据格式定义

gRPC - 组件间数据交换

Eigen - 包括线性代数，矩阵，向量操作，数值解决和其他相关的算法的C++模板库。

SWIG - 一个可以让你的C++代码链接到JavaScript，Perl，PHP，Python，Tcl和Ruby的包装器/接口生成器

Thread Safety Analysis -

MLIR - 全称是Multi-Level Intermediate Representation compiler infrastructure， 编译器的编译器，meta compiler. 其目的是为了在机器学习的前端和后端之间建立起一个中端的IR bridge，来减少前端直接打到后端所涉及到重复建设

wheel - 

MKL - Intel出的数学计算库Math kernel library（MKL)

GEMM - 线性代数库。