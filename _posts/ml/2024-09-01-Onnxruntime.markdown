---
layout: post
title:  "ONNXRuntime生态"
subtitle: "ONNXRuntime"
categories: [MachineLearning]
---


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



# 算子按开发模式分类

* onnxruntime/onnxruntime/core/providers 
* onnxruntime/onnxruntime/contrib_ops
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
