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

* tensorflow/core  核心代码由C++实现。
  * core/ops/ contains the "signatures" of the operations
  * core/kernels/ contains the "implementations" of the operations (including CPU and CUDA kernels)
  * core/framework/ contains the main abstract graph computation and other useful libraries
  * core/platform/ contains code that abstracts away the platform and other imported libraries (protobuf, etc)
* tensorflow/contrib



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



# Runtime

common_runtime

distributed_runtime


# 通信原理
跨节点通信是Tensorflow实现图分布式、可伸缩性的基础，这里节点是个广义含义。通信往往也是分布式的瓶颈(跟计算相比而言)。

## 跨节点如何通信
* 同一主板下的CPU-GPU通信
  * CUDA
    * cudaMemcpy( void* dst, const void* src, size_t count, cudaMemcpyKind kind )  
    * DMA：GPU直接访问主机内存(pinned)，数据复制无需CPU去做。CUDA架构下主机内存(host memory)的2类状态，后者就是DMA方式直接访问的部分。
      * 可分页内存(pageable)： 由OS malloc分配
      * 页锁定内存(pinned)： 由cudaMallocHost分配，OS不对其做分页、交换，始终驻留在物理内存里。
    * tensorflow/core/common_runtime/gpu/ 
      * 显存通信： GPUUtil
      * 显存分配： GPUcudaMallocAllocator、GPUBFCAllocator、PoolAllocator
  * OpenCL & SYCL, CUDA的平行方案
    * OpenCL: 并行访问各类计算资源(CPU/GPU/FPGA...)的低级语言 
    * SYCL: 基于OpenCL包装的高级抽象层，覆盖OpenCL的所有功能，也支持穿插原生OpenCL代码。
    * tensorflow/core/common_runtime/sycl
* 网络通信
  * gRPC 
    * gRPC over TCP-IP，万兆以太网
    * tensorflow/core/distributed_runtime/rpc/ 
      * GrpcServer: 基于 grpc::Server 的服务器，Start/Join/Stop接口，对外体现New/Start/Stop等状态。基于gRPC的异步事件管理机制提供了自己的调用类与标记类，并实现了一套基 于状态转移与回调函数的服务器端调用处理流程。  
        * 依据cluster_spec构建各个节点上的服务  ::grpc::ServerBuilder builder; 
          * AddListeningPort() → RegisterService() → AddCompletionQueue() → BuildAndStart()
      * Call 对象:  tensorflow/core/distributed_runtime/rpc/grpc_call.h    Call<Service, GrpcService, Req, Resp>         TODO
      * 构建 grpc::Channel 维护成 GrpcChannelCache ，   typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;
      * 同步通信  grpc::MasterService::NewStub,  grpc::MasterService::Stub,   grpc::BlockingUnaryCall
      * 异步通信  grpc::WorkerService::AsyncService
* RDMA
  * 专用网络.	通过Kernel bypass等方式，用户态接口可以直接向RDMA网卡发起数据收/发操作，而不需要再进入内核，降低数据传输中的延迟和CPU消耗。
  * RDMA-enabled TensorFlow 
    * tensorflow/contrib/verbs  
    * 仍然需要grpc做控制管理，创建出RDMA channel、RDMA connection
    * RDMA消息体：|type|name_size|name|step_id|buffer_size|remote_addr|rkey|is_dead|data_type|tensor_shape|tensor_bytes|tensor_buffer|
    * IB Verbs的通信原语更加底层，不提供消息缓存、内存分配、动态调度等高级功能，contrib/verbs 通信模块提供了一套基于缓冲区预分配和消息交互的解决方案。
  * RDMA技术被3种网络协议所支持:
    * 1.Infiniband(IB):支持RDMA的新一代网络协议。由于这是一种新的网络技术，因此需要使用RMDA专用网卡与交换机，从硬件级别全面支持RDMA。
    * 2.RDMA OVER Converged Ethernet(RoCE):基于现有的以太网实现RDMA，底层网络包头部是普通的以太网包头，上层网络包头部是Infiniband包头，因此RoCE只需要专有的支持RoCE网卡，就可以在标准以太网基础架构(交换机)上实现RDMA。
    * 3.Internet Wide Area RDMA Protocol(iWARP): iWARP直接将RDMA实现在了TCP上，这允许在标准以太网基础架构(交换机)上使用RDMA，这种方案需要配置支持iWARP的特殊网卡。
* GPUDirect RDMA
  * GPU-memory直接通信RDMA网卡，而不需要CPU-memory去做。
  * tensorflow/contrib/gdr/

## 谁和谁需要通信

### 关于通信需求的铺垫
* C-S模式分工： 
  * Server: tf.distribute.Server、tf.train.Server实例作为每个节点上都有的服务(S,  前面讲过的GrpcServer)，绑定以下两个service： 
    * tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h 
    * Master service 负责划分子图并派发Task给worker service. 
      * 一个TensorFlow集群中只有一个Master Service在真正工作       service MasterService { ... } 
        * rpc CreateSession(CreateSessionRequest) returns (CreateSessionResponse); 
        * rpc DeregisterGraph(DeregisterGraphRequest) returns (DeregisterGraphResponse);
        * rpc RunStep(RunStepRequest) returns (RunStepResponse);
      * Master 负责将 Client Graph 按照任务的名称分裂 (SplitByTask) 为多个 Graph Partition
    * Worker service 负责子图运算 
      * 每个worker都会有 Worker Service工作：    service WorkerService {...} 
        * rpc CreateWorkerSession(CreateWorkerSessionRequest) returns (CreateWorkerSessionResponse);
        * rpc RunGraph(RunGraphRequest) returns (RunGraphResponse);
        * rpc RecvTensor(RecvTensorRequest) returns (RecvTensorResponse)
      * 每个 Worker 对应一个 Graph Partition
  * Client: 一般是python程序，创建tf::session和graph 
    * Client 执行 Session.run 时，传递整个GraphDef给 Master
    * Session是 Client 和 Master的桥梁， Client 通过 GrpcSession 调用 Master Service。  
    * GrpcSession 是 tensorflow::grpc::MasterService 的简单封装。其使用远程设备集作为计算资源，使用 grpc 作为远端调用机制，让调用者在远端设备上对 TensorFlow 图进行计算。 
      * 分布式场景(SessionOptions的target以grpc://开头)  SessionFactory::Register("GRPC_SESSION", new GrpcSessionFactory());       → GrpcSession::Run
      * 单机场景(Client和Master在同一进程里)     SessionFactory::Register("DIRECT_SESSION", new DirectSessionFactory());   →  DirectSession::Run
  * tf.train.Supervisor 、 tf.compat.v1.train.Supervisor、tf.compat.v1.train.MonitoredTrainingSession  
    * 封装了a Coordinator, a Saver, and a SessionManager
    * 分布式环境中可能存在多个 worker 节点，但是其中需要有一个作为 chief worker 节点。chief 需要负责初始化整个运行图，其他 worker 节点将从 chief 节点获取计算图的信息

### TF通信需求
* 进程内的节点通信 
  * 同一CPU内存内部：指针传递即可
  * 跨设备内存的数据(tensor)复制
* 进程间的节点通信 worker replicas 
  * 控制通信： 
    * CS前后端模式下的通道管理、Session管理、图注册、图执行控制
  * 数据通信：  
    * PS拓扑模式下： PS和Worker之间需要通信参数
    * allreduce拓扑模式下：worker之间需要通信参数


## 通信的数据体

*	Tensor对象：主要是weights、gradients 
  * 逻辑上有传输语义但src-dst都在同一空间时，只传输指针
  * src-dst不在同一空间时，才拷贝传输数据本身
* Tensor对象和内部的TensorBuffer对象：始终保存在CPU主内存 
  * Tensor和TensorBuffer都不是真实数据
* TensorBuffer内部指针所指向的数据本身：可能在CPU主内存、也可能在GPU显存。 
  * 跨设备传递  CopyTensor::ViaDMA  
    * Tensor::CopyFromInternal()   在CPU主内存内部拷贝 
      * 直接重置指针TensorBuffer* buf_
    * Tensor::CopyDeviceTensorToCPU()    从GPU内存到CPU内存 
      * GPUUtil::CopyGPUTensorToCPU
    * Tensor::CopyCPUTensortoDevice()    从CPU内存到GPU内存 
      * GPUUtil::CopyCPUTensorToGPU
    * GPUUtil::DeviceToDeviceCopy()   在不同GPU内存之间
      * GPUUtil::SetProtoFromGPU()   由GPU内的tensor生成CPU中的TensorProto
  * 以上涉及GPU的通信时，都没有简单地使用CUDA API (如cudaMemcpy)，而是调用并行框架 StreamExecutor 提供的内存复制机制 。 
    * StreamExecutor是Google各个项目里公用的一个并行编程库，对CUDA/OpenCL编程做了统一封装，直接基于 CUDA 驱动层接口实现，并提供异步复制和完成回调功能
    * StreamExecutor为TensorFlow的执行层面提供了较为统一的抽象，而在底层各种Device的执行管理细节却完全不同。stream_executor下面有cuda和host两个子目录，他们分别是GPU执行引擎和CPU执行引擎所使用的子模块。
*	网络传递      
  * TensorProto →  Tensor 转换： 
    * TensorResponse类用于承接rpc接收的Tensor封装数据    tensorflow/core/distributed_runtime/tensor_coding.h   
    * MakeTensorFromProto() 有CPU/GPU/SYCL多个实现版本
  * Tensor →  TensorProto 转换： 
    * AsProtoField
    * AsProtoTensorContent 
      * EncodeTensorToByteBuffer  在GrpcRecvTensorAsync使用:  将有可能保存在不同种类设备内存上的tensor数据以尽可能减少内存分配与复制次数的方式写入 ByteBuffer 对象，供gRPC发送

## 通信模型
在TensorFlowAPI层面，通信接口对用户不直接可见，runtime框架(common or distributed)根据用户代码里的集群和设备配置，在DAG构建时自动插入通信操作。

###	统一的消息传递模型
* Rendezvous（会合点）机制： 英文单词的意思a meeting at an agreed time and place。   以会合点为中心的生产者-消费者模型，一致的异步收发协调机制 
  * 唯一标识符ParsedKey:       [src_device];[src_incarnation];[dst_device];[tensor_name];[frame_id]:[iter_id] 
    * src_device：消息发送源的字符串信息，形如/job:localhost/replica:0/task_id:0/device:GPU:0
  * Rendezvous实现类的实现策略大体是：有自己的消息队列缓存，发送(Send)和接收(Recv)通过Table完成。 
    * Table:    typedef std::deque<Item*> ItemQueue;     typedef gtl::FlatMap<uint64/*ParsedKey*/, ItemQueue> Table;     
    * Value：这就是参与通信Tensor本体
    * Waitor：这是在确认Tensor被接收端完成接收后的处理函数，也就是consumer处理该Tensor的函数过程
*	Rendezvous  生产者-消费者传输Tensor过程的抽象类 
  *	tensorflow/core/framework/rendezvous.h 为不同通信场景下使用的子类提供统一的接口 
    * Send() 同步发送
    * RecvAsync() 异步接收
    * Recv() 同步接收（基于RecvAsync封装
    * StartAbort() 取消操作
    * DoneCallback   由tensor消费方提供的回调， 在tensor ready后调用
*	LocalRendezvousImpl 本地设备通信会合点的实现类 
  *	tensorflow/core/framework/rendezvous.cc   实现了与收发操作顺序无关的本地消息传递过程。在Send和RecvAsync顺序相对异步的情况下，waitor函数的执行时机只有两种情况，它取决于Send的供给和RecvAsync的需求哪一个先到达。 
  	* 若生产者先到达，那么waiter函数的调用由RecvAsync执行。
  	* 若消费者的需求先到达，那么waiter函数的调用由Send执行。简而言之，总是迟到的一方执行waiter函数。
*	IntraProcessRendezvous  本机同进程跨设备通信会合点的实现类
  *	tensorflow/core/common_runtime/rendezvous_mgr.h 
  *	考量Tensor的生产方和消费方是存在于CPU还是GPU，是否可以通过P2P直接拷贝，还是需要通过Host做中转
  *	CopyTensor::ViaDMA
*	BaseRemoteRendezvous 
  * 跨进程通信的各种 Rendezvous 都需要依据自己不同的协议来实现，所以在 RemoteRendezvous 和真正特化的四种 Rendezvous 中间加入了一个中间层 BaseRemoteRendezvous，这个类起到了承上启下的作用，提供了公共的 Send 和 Recv 方法，尽可能复用代码。 
  * RpcRemoteRendezvous
  * RDMARemoteRendezvous  网卡需要支持RDMA协议
  * GdrRemoteRendezvous.   GPUDirect RDMA
  * MPIRemoteRendezvous.  由MPI over infiniband负责tensor的跨进程通信
*	RpcRemoteRendezvous  分布式通信会合点的实现类 
  *	tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc   TODO
	* send/recv 的数据传输是通过 WorkerInterface 的派生类作为接口完成，底层就是WorkerService是RPC接口 
	* rpc RecvTensor(RecvTensorRequest) returns (RecvTensorResponse) 
  	* TensorResponse类是 Tensor 类的封装形式，将保存在 gRPC消息中的 TensorProto 对象反序列化为 Tensor 对象 
  	* 实现类 RpcRecvTensorCall 对grpc的抽象做了一层封装管理其生命周期，继承gRPC通信实体的抽象类BaseRecvTensorCall
  *	Rec/Send的行为： Recv方可以认为是Client，Send方可以认为是Server，通过发送Request和Response来完成Tensor的传输。 
    * Recv方是Tensor的接收方，它将所需要的Tensor对应的ParsedKey拼出后，主动向Send方主动发出Request。
    * Send过程并不涉及跨进程传输，只是将Ready的Tensor挂入本地Table之中。
    * Send方在接收到Request后立即在本地Table中查找所需要的Tensor，找到后将Tensor封装成Response发送回Recv方。

### 统一的编程接口
*	通信过程都封装成了OpKernel.    tensorflowIcore/kerneIs/sendrecv_ops.h 
  *	SendOp 发送接口集成自OpKernel并实现Compute方法，是一个同步操作 
  *	RecvOp 接收接口继承自AsyncOpKernel并实现ComputeAsync方法，是一个异步操作 
  *	OpKernelContext中可以用会合点基类指针指向不同类型的子类对象，以多态方式调用不同场景的通信实现
  *	ArgOp和RetvalOp 广义上讲也是通信OP，它相当于本地模式版本的RecvOp/SendOp

### 全图构建时有关通信的内部工作
* 总体的目标是要在图构建时完成图分割并自动(隐式)插入通信操作 
  * 第一步：Master根据Client提交的GraphDef 构建出 Full Graph， 然后根据本次fetches和feeds遍历，剪枝生成本次运行的最小依赖子图 ClientGraph。
  * 第二步：Full Graph 在 Master 中首次分裂SplitByWorker， 生成各个worker的 PartitionGraph
  * 第三步：在 Worker 中二次分裂SplitByDevice
* 图的组成： 
  * node: 
    * PlaceholderOp
    * VariableOp
    * ConstantOp
    * 计算 op
  * edge: 
    * tensor
*	剪枝、子图提取过程： 
  * 根据Graph的输入输出列表，反向遍历全图，找到几个最小依赖的子图，从而方便并行计算。
  * 子图提取时，添加通信操作，切分之后的一对数据收发操作等价于切分之前的一条边。
  * tensorflow/core/graph/subgraph.cc 
    * RewriteGraphForExecution()  根据输入输出feed fetch，对graph进行增加节点或删除节点等操作
    * FeedInputs(). 增加输入节点 
    * FetchOutputs(). 增加输出节点 
    * PruneForTargets().  广度优先搜索最小依赖子图
* 图分割依据： 
  * NodeToLocFunc node_to_loc  
    * DeviceNameUtils::SplitDeviceName()
    * node→assigned_device_name()
  * tf.compat.v1.train.replica_device_setter
    * with tf.device()语句可以指明设备，如果不慎将variable放到普通节点而非ps节点的话，势必造成训练问题。因而提供了该方法，告诉worker节点默认在本机分配Op，且自动把所有Variables分配到ps节点。
    * 由于ps往往不止一个，这个函数在为各个Variable分配ps时默认采用简单的round-robin方式，就是按次序将参数挨个放到各个ps上，但这个方式可能不能使ps负载均衡，如果需要更加合理，可以采用tf.contrib.training.GreedyLoadBalancingStrategy策略。
*	按Task/Worker图拆分过程 SplitByWorker：
  * 用户代码主动分配
  * 默认策略自动分配： 
    * PS模式下variable分配   
*	按设备拆图过程  SplitByDevice： 
  *	CPU/GPU拆分 
  *	消除跨设备的边，并检查同一个Tensor在src-dst是否有重复的边。 
  *	tensorflow/core/graph/graph_partition.cc 最核心的Partition()方法

### Runtime时有关通信的内部工作
*	MasterSession::Run →  DoRunWithLocalExecution 
  * ReffedClientGraph::RunPartitions   执行1个step 
    * Worker::RunGraphAsync 
      * Worker::DoRunGraph 
        * worker_session ->graph_mgr-> GraphMgr::ExecuteAsync 
          * SendTensorsToRendezvous 
            * Rendezvous::ParseKey
            * rendezvous->Send
        * 回调  worker_session ->graph_mgr->RecvOutputs  
          * RecvOutputsFromRendezvous 
            * Rendezvous::ParseKey
            * rendezvous->Recv
        * GraphMgr::StartParallelExecutors 
          * Executor in ThreadPool

## 通信的步调
参数通信的目的是完成完整的机器(参数)学习任务，那么通信的步调如何选择（既快又好）。
*	同步和异步的拓扑：
  * 同步： 
    * MirroredStrategy
    * MultiWorkerMirroredStrategy
    * horovod 
  * 异步： 
    * ParameterServerStrategy
* 同步和异步的最优化算法
  * 同步通信的优化方法 Bulk Synchronous Parallel, BSP： 
    * BSP-SGD、模型平均、ADMM、EA-SGD
  * 异步通信的优化方法 Asynchronous Parallel, ASP： 
    * ASGD、HogWild!、Cyclades
    * AdaptiveRevision、AdaDelay、DC-SGD
*	延时同步并行 Stale Synchronous Parallel, SSP： 
  * 控制最快和最慢节点之间相差的迭代次数不超过阈值



# 梯度是如何计算的？

我们要在Tensorflow Graph中计算梯度，当 Session 运行起来的时候，TF会自动按图运算流向的反方向生产一张逆向的图。

我们自己创建的图叫做 Forward Pass Graph， 它由输入数据(placeholder)、运算单元(OP)、和模型参数（Variables）构成。而TF自动生成的对应的图叫做 Backward Pass Graph。

## GradientDef

GradientDef 定义了一个函数对应的梯度函数。















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

* Protobuffer - 数据格式定义
* gRPC - 组件间数据交换
* Eigen - 包括线性代数，矩阵，向量操作，数值解决和其他相关的算法的C++模板库。
* SWIG - 一个可以让你的C++代码链接到JavaScript，Perl，PHP，Python，Tcl和Ruby的包装器/接口生成器
* Thread Safety Analysis -
* MLIR - 全称是Multi-Level Intermediate Representation compiler infrastructure， 编译器的编译器，meta compiler. 其目的是为了在机器学习的前端和后端之间建立起一个中端的IR bridge，来减少前端直接打到后端所涉及到重复建设
* wheel - 
* MKL - Intel出的数学计算库Math kernel library（MKL)
* GEMM - 线性代数库。
* sycl
* rdma