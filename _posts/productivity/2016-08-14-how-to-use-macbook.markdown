---
title: "Macbook使用技巧"
subtitle: ""
categories: [productivity]
layout: post
---
使用Macbook作为日常工作和生活的笔记本电脑，在硬件方面是看中它的小巧轻便，续航能力强，纯固态硬盘带来的高速体验；在软件方面是看中它和Linux的亲缘关系。使用Macbook有几年了，这里分享一下我的一些使用心得。


# 不可或缺的工具

## Alfred

效率工具。它将搜索引擎搜索、本地文件查找、本地App启动、本地定制化命令操作、计算器、词典等功能合为一个入口。不管做什么事情，只需要一个快捷键启动这个入口，就能查询想要的东西。

## Moom

效率工具。以往我们总是使用鼠标的拖拽、拉伸窗口来控制一个窗口的布局，使用Moom设置快捷键来将这些重复性的工作归纳为快捷键吧。

## Homebrew

工具包管理器，相当于Ubuntu的apt-get，Redhat的yum。

其他的App比如浏览器、终端、文本编辑工具、IDE、Email工具、科学上网工具、播放器，想必大家都知道了。


# 使用外接鼠标

MacOS对于鼠标滚轮的设计和触摸板的滑屏设计一致，与WindowsOS的设计相反。
对于触摸板来讲，Mac的滑屏设计是我更认同的，毕竟和触摸形式一致。而鼠标滚轴来讲，我更接受Windows的方向。
在用鼠标滚轮滚动屏幕时总是不习惯，第一反应总是滚错方向。如果把鼠标设置的滚轮方向修改为非Natural的，那么触摸板的方向也会随之改变。解决办法是使用软件Karabiner，同时勾选"Don't remap Apple's pointing devices"和“Reverse Vertical Scrolling”。这样子触摸板还是Mac的模式，而鼠标采用Windows的模式。

# 使用外接键盘

Macbook内置的键盘其实挺好的，有时我也会连接HHKB寻求更高体验。把外置键盘直接搁在内置键盘上可能会不小心敲击到内置键盘的某个按键，所以最好是将内置键盘关掉。方法是使用软件Karabiner，勾选“Disable an internal keyboard while external keyboards are connected.”。

# 使用外接显示器

## 使用双显示器工作

插上外接显示器，选择扩展屏幕模式，就能使用双屏来工作了。

## 只使用外接显示器工作

这个工作场景是把Macbook当台式机主机来用，使用外接显示器、外接键盘、外接显示器。默认的话插上电源、插上外接显示器，只要合上笔记本盖子，显示模式就是只用外接显示器。但一个问题就是合上盖子后本子的散热问题，这本不会是什么大问题，毕竟合上本子之后屏幕连接处还是能出风的。但和我一样龟毛的人可能会觉得键盘的散热功能不能发挥了。一个妥协方案就是用一个磁性物质键盘左侧放在TAB和CAP键之间（我使用的是一个磁性很弱的小冰箱贴），于是开着盖子就能让屏幕、内置键盘、触摸板全部关闭。



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