---
title: "NVIDIA Video SDK 技术解析"
categories: [design]
layout: post
---

随着GPU计算的普及，NVIDA显卡为视频处理、图像处理、虚拟现实、机器学习提供了丰富的的SDK。具备NVIDA显卡硬件和NVIDIA驱动之后，就可以使用这些SDK。

# NVIDIA Capture SDK

这套以前叫做Grid SDK。它提供给使用者获取屏幕或应用的图像的功能。调用者可以获取桌面屏幕buffer，也可以获取某个应用所渲染的区域。可以以图像的形式，也可以以视频流的形式。


## NvFBC

NVFBC的全称是NVIDIA Frame Buffer Capture，用来抓取OS全屏帧buffer。它把全屏图像数据放到一个GPU buffer。对于Windows来讲，不管 Windows Desktop Manager是否启动，NVFBC都可以工作。这个buffer在内部可以直接通过NVENCODE API直接编码为H.264/HEVC。


## NvIFR

NvIFR的全程是NVIDIA Inband Frame Readback，所以它是用来抓取应用程序所渲染的区域。


# NVIDIA Video Codec SDK


NVIDIA GPU 包含了基于硬件的H.264/HEVC视频编码器，称作是NVENC。NVENC硬件以YUV/RGB作为输入，输出H.264/HEVC标准的视频比特流。

## NVENCODE API

nvEncodeAPI.dll for Windows 和 libnvidia-encode.so for Linux 跟随NVIDIA驱动安装上，其中提供了NVENCODE API。

下面是使用NVIDIA video encoder API的一般流程：

		初始化编码器。
		设置编码参数。
		分配输入、输出buffer。
		将frame写入input buffer。
		从output buffer读出视频比特流。 写入和读出的过程可以是同步的，在Window7以上的Windows版本可以是异步的。
		关闭回话。
		释放输入、输出buffer。


## NVDECODE API

NVDEC 是NVIDIA的硬件加速H264解码器。能够解码H.264, HEVC (H.265), VP8, VP9, MPEG-1, MPEG-2, MPEG-4，VC-1等格式。NVDEC将压缩的视频流解码，得到YUV帧数据，将其拷贝到内存中，用于播放或转码。


下面是视频播放的一般流程：

		解析输入的视频源。
		查询解码能力。
		使用NVDECODE API解码视频比特流。
		将解码得到的YUV格式转换为RGBA。
		将RGBA Surface映射为DirectX 9.0/OpenGL Texture。
		将Texture画到屏幕上。


# 参考链接

https://developer.nvidia.com/capture-sdk

https://developer.nvidia.com/nvidia-video-codec-sdk


<!--
这里是注释区

```
print "hello"
```

***Stronger***

{% highlight python %}
print "hello, Lucky!"
{% endhighlight %}

![My image]({{ site.baseurl }}/images/emule.png)

My Github is [here][mygithub].
[mygithub]: https://github.com/lucky521

-->