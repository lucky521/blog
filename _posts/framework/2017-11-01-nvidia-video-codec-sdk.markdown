---
title: "NVIDIA Video SDK 技术解析"
categories: [framework]
layout: post
---

随着GPU计算的普及，NVIDA显卡为视频处理、图像处理、虚拟现实、机器学习提供了丰富的的SDK。具备NVIDA显卡硬件和NVIDIA驱动之后，就可以使用这些SDK。

# NVIDIA Capture SDK

这套以前叫做Grid SDK。它提供给使用者获取屏幕或应用的图像的功能。调用者可以获取桌面屏幕buffer，也可以获取某个应用所渲染的区域。可以以图像的形式，也可以以视频流的形式。

Capture SDK提供了两种API，NvFBC适合提供远程桌面方案，NvIFR适合截取特定应用的画面。

Shadowplay 是Nvidia显卡驱动提供的游戏录屏工具，能够记录下过去一段时间的屏幕视频。

## NvFBC

NVFBC的全称是NVIDIA Frame Buffer Capture，用来抓取OS全屏帧buffer。它把全屏图像数据放到一个GPU buffer。对于Windows来讲，不管 Windows Desktop Manager是否启动，NVFBC都可以工作。这个buffer在内部可以直接通过NVENCODE API直接编码为H.264/HEVC。

NVIDIA提供了5种不同的NvFBC接口：
	
		NVFBC_TO_SYS 捕获桌面将frame拷贝到系统内存。
		NVFBC_TO_CUDA 捕获桌面拷贝到GPU上的CUDA内存。
		NVFBC_TO_GL 捕获桌面将其拷贝到OpenGL texture。
		NVFBC_TO_DX9VID 捕获桌面并将其拷贝到DX9 surface。
		NVFBC_TO_HW_ENCODER 捕获桌面，使用硬件编码器编码城H264/HEVC流，然后将其拷贝到系统内存。




## NvIFR

NvIFR的全程是NVIDIA Inband Frame Readback，所以它是用来抓取应用程序所渲染的区域。



## NvFBC 用法

```cpp
#include <NvFBCLibrary.h>
#include <NvFBC/nvFBCHwEnc.h>


    NvFBCLibrary nvfbcLibrary;
    NvFBCToSys *nvfbcToSys = NULL;

    DWORD maxDisplayWidth = -1, maxDisplayHeight = -1;
    BOOL bRecoveryDone = FALSE;

    NvFBCFrameGrabInfo grabInfo;
    unsigned char *frameBuffer = NULL;
    unsigned char *diffMap = NULL;
    char frameNo[10];
    std::string outName;

    //! Load NvFBC
    nvfbcLibrary.load();

    //! Create an instance of NvFBCToSys
    nvfbcToSys = (NvFBCToSys *)nvfbcLibrary.create(NVFBC_TO_SYS, &maxDisplayWidth, &maxDisplayHeight);

    NVFBCRESULT status = NVFBC_SUCCESS;

    //! Setup the frame grab
    NVFBC_TOSYS_SETUP_PARAMS fbcSysSetupParams = {0};
    fbcSysSetupParams.dwVersion = NVFBC_TOSYS_SETUP_PARAMS_VER;
    fbcSysSetupParams.eMode = args.eBufFormat;
    fbcSysSetupParams.bWithHWCursor = args.bHWCursor;
    fbcSysSetupParams.bDiffMap = FALSE;
    fbcSysSetupParams.ppBuffer = (void **)&frameBuffer;
    fbcSysSetupParams.ppDiffMap = NULL;

    status = nvfbcToSys->NvFBCToSysSetUp(&fbcSysSetupParams);
    if (status == NVFBC_SUCCESS)
    {
        //! Sleep so that ToSysSetUp forces a framebuffer update
        Sleep(100);
        
        NVFBC_TOSYS_GRAB_FRAME_PARAMS fbcSysGrabParams = {0};
        //! For each frame to grab..
        for(int cnt = 0; cnt < args.iFrameCnt; ++cnt)
        {
            outName = args.sBaseName + "_" + _itoa(cnt, frameNo, 10) + ".bmp";
            //! Grab the frame.  
            // If scaling or cropping is enabled the width and height returned in the
            // NvFBCFrameGrabInfo structure reflect the current desktop resolution, not the actual grabbed size.
            fbcSysGrabParams.dwVersion = NVFBC_TOSYS_GRAB_FRAME_PARAMS_VER;
            fbcSysGrabParams.dwFlags = args.iSetUpFlags;
            fbcSysGrabParams.dwTargetWidth = args.iWidth;
            fbcSysGrabParams.dwTargetHeight = args.iHeight;
            fbcSysGrabParams.dwStartX = args.iStartX;
            fbcSysGrabParams.dwStartY = args.iStartY;
            fbcSysGrabParams.eGMode = args.eGrabMode;
            fbcSysGrabParams.pNvFBCFrameGrabInfo = &grabInfo;
        
            status = nvfbcToSys->NvFBCToSysGrabFrame(&fbcSysGrabParams);
            if (status == NVFBC_SUCCESS)
            {
                bRecoveryDone = FALSE;
                //! Save the frame to disk
                switch(args.eBufFormat)
                {
                case NVFBC_TOSYS_ARGB:
                    SaveARGB(outName.c_str(), frameBuffer, grabInfo.dwWidth, grabInfo.dwHeight, grabInfo.dwBufferWidth);
                    fprintf (stderr, "Grab succeeded. Wrote %s as ARGB.\n", outName.c_str() );
                    break;

                case NVFBC_TOSYS_RGB:
                    SaveRGB(outName.c_str(), frameBuffer, grabInfo.dwWidth, grabInfo.dwHeight, grabInfo.dwBufferWidth);
                    fprintf (stderr, "Grab succeeded. Wrote %s as RGB.\n", outName.c_str());
                    break;

                case NVFBC_TOSYS_YUV444p:
                    if(args.yuvFile) {
                        fwrite(frameBuffer, grabInfo.dwWidth*grabInfo.dwHeight*3, 1, args.yuvFile);
                    }
                    else {
                        SaveYUV444(outName.c_str(), frameBuffer, grabInfo.dwWidth, grabInfo.dwHeight);
                        fprintf (stderr, "Grab succeeded. Wrote %s as YUV444 converted to RGB.\n", outName.c_str());
                    }
                    break;

                case NVFBC_TOSYS_YYYYUV420p:
                    if(args.yuvFile) {
                        fwrite(frameBuffer, grabInfo.dwWidth*grabInfo.dwHeight*3/2, 1, args.yuvFile);
                    }
                    else {
                        SaveYUV420(outName.c_str(), frameBuffer, grabInfo.dwWidth, grabInfo.dwHeight);
                        fprintf (stderr, "Grab succeeded. Wrote %s as YYYYUV420p.\n", outName.c_str() );
                    }
                    break;

                case NVFBC_TOSYS_RGB_PLANAR:
                    SaveRGBPlanar(outName.c_str(), frameBuffer, grabInfo.dwWidth, grabInfo.dwHeight);
                    fprintf (stderr, "Grab succeeded. Wrote %s as RGB_PLANAR.\n", outName.c_str() );
                break;

                case NVFBC_TOSYS_XOR:
                    // The second grab results in the XOR of the first and second frame.
                    fbcSysGrabParams.dwVersion = NVFBC_TOSYS_GRAB_FRAME_PARAMS_VER;
                    fbcSysGrabParams.dwFlags = args.iSetUpFlags;
                    fbcSysGrabParams.dwTargetWidth = args.iWidth;
                    fbcSysGrabParams.dwTargetHeight = args.iHeight;
                    fbcSysGrabParams.dwStartX = 0;
                    fbcSysGrabParams.dwStartY = 0;
                    fbcSysGrabParams.eGMode = args.eGrabMode;
                    fbcSysGrabParams.pNvFBCFrameGrabInfo = &grabInfo;
                    status = nvfbcToSys->NvFBCToSysGrabFrame(&fbcSysGrabParams);
                    if (status == NVFBC_SUCCESS)
                        SaveRGB(outName.c_str(), frameBuffer, grabInfo.dwWidth, grabInfo.dwHeight, grabInfo.dwBufferWidth);

                    fprintf (stderr, "Grab succeeded. Wrote %s as XOR.\n", outName.c_str() );
                    break;
                case NVFBC_TOSYS_ARGB10:
                    SaveARGB10(outName.c_str(), frameBuffer, grabInfo.dwWidth, grabInfo.dwHeight, grabInfo.dwBufferWidth);
                    fprintf(stderr, "Grab succeeded. Wrote %s as ARGB10.\n", outName.c_str());
                    break;

                default:
                    fprintf (stderr, "Un-expected grab format %d.", args.eBufFormat);
                    break;
                }
            }
        }
    }
    
    if (nvfbcToSys)
    {
        //! Relase the NvFBCToSys object
        nvfbcToSys->NvFBCToSysRelease();
    }

    if(args.yuvFile) {
        fclose(args.yuvFile);
    }

    return 0;

```



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