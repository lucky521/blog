---
layout: post
title:  "ONNXRuntime生态"
subtitle: "ONNXRuntime"
categories: [MachineLearning]
---


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