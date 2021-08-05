---
title: "如何使用XBox游戏手柄?"
categories: [framework]
layout: post
---

![Xbox controller](http://compass.microsoft.com/assets/55/69/556932e3-5fb4-40e9-a868-450606fd1a8c.jpg?n=pop1.jpg)

Xbox游戏手柄是业界有名的游戏手柄，它可以在Xbox、Windows、Mac、其他游戏平台等使用，最近花了一点时间关注了一下相关的内容。

# 几点疑问

- xbox的每个按键对应的是OS的什么事件？

XInputGetState这个API可以主动获取手柄的全部按键状态，返回的是XINPUT_STATE结构体。
XInputGetKeystroke这个API可以主动获取到一个手柄事件(KEYSTROKE)，返回的是XINPUT_KEYSTROKE结构体。

- xbox震动对应的是OS的什么事件？

XInputSetState这个API可以让手柄震动。

- 怎么让OS被动获取到手柄的事件？

这是不行的。其实普通键盘鼠标的事件也不是OS被动获取的，都是windows事件循环中轮询的。在事件循环加一个timer，不停的轮询手柄按键检测函数吧。

- 游戏使用游戏手柄是不是有两种方式？

一种是API直接处理手柄的按键事件，一种是有个驱动将手柄按键事件转换为键盘事件。
游戏或者普通程序是游戏手柄的消费者，所以是它们的代码在获取手柄的按键数据。
要么游戏是调用Xinput API获取到手柄事件，要么直接监听这些virtual-key-code的事件。



# 微软平台的系统标准接口    

## Windows API XInput

https://msdn.microsoft.com/en-us/library/windows/desktop/hh405053(v=vs.85).aspx
XInput Game Controller API enables applications to receive input from the Xbox 360 Controller for Windows.
函数非常少，只有8个接口函数。
https://msdn.microsoft.com/en-us/library/windows/desktop/ee417007(v=vs.85).aspx

Xinput example demo
http://www.codeproject.com/Articles/26949/Xbox-Controller-Input-in-C-with-XInput

## 手柄按键的类型

C:\Program Files\Microsoft SDKs\Windows\v7.0A\Include\Xinput.h

```
    结构体 XINPUT_GAMEPAD 能够展示出手柄上所有按键的状态
    typedef struct _XINPUT_GAMEPAD {
      WORD  wButtons; //这是一个Bitmask，展示了14个简单按键的状态。
      BYTE  bLeftTrigger;
      BYTE  bRightTrigger;
      SHORT sThumbLX;  //左右两个stick圆盘用x,y坐标值来表示状态。
      SHORT sThumbLY;
      SHORT sThumbRX;
      SHORT sThumbRY;
    } XINPUT_GAMEPAD, *PXINPUT_GAMEPAD;
```

## 手柄按键的键值

scan-code  硬件手柄每一个按键自己所产生的值。 （这是手柄厂家设计确定的。）

virtual-key-code   操作系统OS所理解的一个按键的值。参见WinUser.h。（这是操作系统厂家设计确定的。）

一定会有一个驱动程序，将scan-code映射到virtual-key-code。这样就实现了不同厂家游戏手柄来玩同一个游戏。

## USB - HID设备

https://msdn.microsoft.com/en-us/library/windows/desktop/hh405052(v=vs.85).aspx




# 苹果平台的系统标准接口

和win平台的xinput一样，苹果平台的开发者如果想让自己的游戏能够使用手柄，也需要使用苹果操作系统的接口。

https://developer.apple.com/library/content/documentation/ServicesDiscovery/Conceptual/GameControllerPG/Introduction/Introduction.html


## 说明文档

https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinput_keystroke(v=vs.85).aspx

## GameController

https://developer.apple.com/reference/gamecontroller




# 相关工具

Xpadder

JoyToKey

TattieBogle

https://github.com/360Controller/360Controller

https://github.com/x360ce/x360ce

Emulator Enhancer http://www.bannister.org/software/ee.htm

https://yukkurigames.com/enjoyable/




# 针对特定游戏的优化设计

## Dead Zone 死区

http://www.paopaoche.net/tv/89686.html





