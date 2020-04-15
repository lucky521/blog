---
title: "Macbook使用技巧"
subtitle: ""
categories: [productivity]
layout: post
---
使用Macbook作为日常工作和生活的笔记本电脑，在硬件方面是看中它的小巧轻便，续航能力强，纯固态硬盘带来的高速体验；在软件方面是看中它和Linux的亲缘关系。使用Macbook有几年了，这里分享一下我的一些使用心得。


# 不可或缺的软件工具

## Iterm2

尤其是其中配置的triggers，不能丢。

## Alfred

效率工具。它将搜索引擎搜索、本地文件查找、本地App启动、本地定制化命令操作、计算器、词典等功能合为一个入口。不管做什么事情，只需要一个快捷键启动这个入口，就能查询想要的东西。

## Moom

效率工具。以往我们总是使用鼠标的拖拽、拉伸窗口来控制一个窗口的布局，使用Moom设置快捷键来将这些重复性的工作归纳为快捷键吧。

## Homebrew

工具包管理器，相当于Ubuntu的apt-get，Redhat的yum。

其他的App比如浏览器、终端、文本编辑工具、IDE、Email工具、科学上网工具、播放器，想必大家都知道了。




# 如果使用外接鼠标

MacOS对于鼠标滚轮的设计和触摸板的滑屏设计一致，与WindowsOS的设计相反。
对于触摸板来讲，Mac的滑屏设计是我更认同的，毕竟和触摸形式一致。而鼠标滚轴来讲，我更接受Windows的方向。
在用鼠标滚轮滚动屏幕时总是不习惯，第一反应总是滚错方向。如果把鼠标设置的滚轮方向修改为非Natural的，那么触摸板的方向也会随之改变。
解决办法1是使用软件Karabiner，同时勾选"Don't remap Apple's pointing devices"和“Reverse Vertical Scrolling”。这样子触摸板还是Mac的模式，而鼠标采用Windows的模式。
解决办法2是使用Scroll Reverser，简单方便。


# 如果使用外接键盘

Macbook内置的键盘其实挺好的，有时我也会连接HHKB寻求更高体验。把外置键盘直接搁在内置键盘上可能会不小心敲击到内置键盘的某个按键，所以最好是将内置键盘关掉。
方法是使用软件Karabiner，勾选“Disable an internal keyboard while external keyboards are connected.”。



# 如果使用外接显示器

## 使用双显示器工作

插上外接显示器，选择扩展屏幕模式，就能使用双屏来工作了。

## 只使用外接显示器工作

这个工作场景是把Macbook当台式机主机来用，使用外接显示器、外接键盘、外接显示器。默认的话插上电源、插上外接显示器，只要合上笔记本盖子，显示模式就是只用外接显示器。但一个问题就是合上盖子后本子的散热问题，这本不会是什么大问题，毕竟合上本子之后屏幕连接处还是能出风的。但和我一样龟毛的人可能会觉得键盘的散热功能不能发挥了。一个妥协方案就是用一个磁性物质键盘左侧放在TAB和CAP键之间（我使用的是一个磁性很弱的小冰箱贴），于是开着盖子就能让屏幕、内置键盘、触摸板全部关闭。




# 清理系统磁盘空间

Mac笔记本往往硬盘不够大。用着用着就不够用了。这时候就要想办法清理空间了。

推荐使用开源工具OmniDiskSweeper扫描一下磁盘，看看每一个目录、文件的磁盘占用情况。


值得清理的地方

    time machine的文件
    sudo tmutil listlocalsnapshots /
    tmutil deletelocalsnapshots 2017-11-27-005359


    anaconda的包文件
    conda clean -p
    conda clean -t


    虚拟机的虚拟磁盘文件
    /Applications/VMware\ Fusion.app/Contents/Library/vmware-vdiskmanager -d "Virtual Disk.vmdk"
    /Applications/VMware\ Fusion.app/Contents/Library/vmware-vdiskmanager -k "Virtual Disk.vmdk"


    VScode Cache
    Press Ctrl + Shift + P -> type command Clear Editor History


# 通过命令行设置电源选项

比如我设置的一个crontab任务：
*/7 * * * * echo "password" | sudo -S  pmset -c displaysleep 60
参考自 https://www.cnblogs.com/zhengran/p/4802582.html


# 尝试纯键盘工作

总有烦躁的时候，觉得鼠标和触摸板的效率都比如键盘快捷键来的快捷，于是尝试各种办法来实现纯键盘工作。



