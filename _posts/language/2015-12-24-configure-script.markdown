---
title: "理解 configure 脚本"
subtitle: "GNU Configure Script"
categories: [Language]
layout: post
---
# configure

configure脚本为了让一个程序能够在各种不同类型的机器上运行而设计的。在使用make编译源代码之前，configure会根据自己所依赖的库而在目标机器上进行匹配。

约定俗成的，所有的configure脚本都把脚本文件名起为`configure`，一般来讲都是shell脚本，根据所在的系统环境生成makefile文件。有时候看到一些程序的configure内容超级之多，让人难以看下去。

configure脚本运行时扫描当前环境，生成一个名为`config.status`的子脚本。子脚本将`Makefile.in`文件转换为适应于当前系统环境的`Makefile`文件。


# GNU build system 

这个工具能够帮助我们生成configure脚本。[GNU build system](https://www.gnu.org/software/automake/faq/autotools-faq.html)，又叫做`Autotools`。

这个工具查询当前系统的环境变量、平台架构、当前所依赖的库以及他们的位置，然后根据这些信息去make，这样就能实现同一套代码仅需configure一下就可以在不同的unix/linux系统中编译运行。

Autotools包含的命令有`autoconf`，`automake`，`libtool`。


## autoconf

autoconf命令的主要作用是创建`configure`。它基于`configure.ac`文件生成`configure`文件。

### autoscan
`autoscan`是autoconf的辅助命令，它创建一个预备的configure（`configure.scan`），作为autoconf命令的输入。我们可以按需在这个文件上进行添加内容，修改之后手工修改名字变成`configure.ac`文件。

### autoheader
`autoheader`也是autoconf的辅助命令，用于生成一个configure的模版头文件`config.h.in`。


## automake

automake命令的主要作用是创建`Makefile`。以`Makefile.am`和`configure.ac`为输入，以`Makefile.in`为输出。`Makefile.in`在configure执行时用于生成`Makefile`。


## libtool

libtool命令的主要作用是创建和使用可移植的共享（动态链接）库。



## 其他相关命令

***`ifnames`*** 命令可以列出程序中的预处理标示符，比如`#if`,`#ifdef`等。

***`m4`*** 处理文件中的宏。

***`pkg-config`*** 查询已安装库的接口。



# configure 标准文件流
下面的命令和文件流清晰的描述了从configure到make的整个过程。

![configure 标准流程](https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Autoconf-automake-process.svg/400px-Autoconf-automake-process.svg.png)

# 参考资料
[configure](https://en.wikipedia.org/wiki/Configure_script)

[GNU build system](https://en.wikipedia.org/wiki/GNU_build_system)

[autotools faq](https://www.gnu.org/software/automake/faq/autotools-faq.html)

[autoconf doc](http://www.gnu.org/software/autoconf/manual/autoconf.html)


<!--
这里是注释区


{% highlight python %}
print "hello, Lucky!"
{% endhighlight %}

![My image]({{ site.baseurl }}/images/emule.png)

My Github is [here][mygithub].

[mygithub]: https://github.com/lucky521


-->