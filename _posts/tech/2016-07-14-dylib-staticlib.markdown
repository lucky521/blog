---
title: "动态链接库和静态链接库"
subtitle: "Dynamic Library and Static Library"
categories: [Tech]
layout: post
---
# 动态库和静态库

- 在Win下，动态库以.dll结尾，静态库以.lib结尾。
- 在Linux下，动态库文件以.so结尾，静态库以.a结尾。
- 在Mac下，动态库以.dylib结尾，静态库以.a结尾。


# 动态库的优势和劣势

- 动态函数库在编译的时候并没有被编译进目标代码中，你的程序执行到相关函数时才调用该函数库里的相应函数，因此动态函数库所产生的可执行文件比较小。由于函数库没有被整合进你的程序，而是程序运行时动态的申请并调用，所以程序的运行环境中必须提供相应的库。 动态函数库的改变并不影响你的程序，所以动态函数库的升级比较方便。
- 某个程序在运行时要调用某个动态链接库函数的时候，OS首先查看所有正在运行的进程，找找看是否已经有人载入了这个库。如果有的话，直接用。如果没有才会载入。这样的第一个优点就是节省内存空间。动态调入意味着是等需要的时候才调入内存，而不是不管用不用都要先放到内存里来。
- 我如果想要升级某个软件。如果升级的位置是在dll里，那软件其他的部位不需要重新编译链接。所以升级方便。

# 静态库的优势和劣势

- 利用静态函数库编译成的文件比较大，因为整个函数库在编译时都会被整合进目标代码中，他的优点就显而易见了，即编译后的执行程序不需要外部的函数库支持，因为所有使用的函数都已经被编译进去了。当然这也会成为他的缺点，因为如果你静态链接的函数库改变了，那么你的程序必须重新编译。
- 代码更精简，因为不必做版本检查。
- 程序分发时文件个数少，因为静态链接到源文件里了。
- 只编译进来库中所用的部分，而不用整个库。
- 生成的binary占空间更大。
- 重复的库可能出现在多个进程，浪费内存。
- 库内部更新的话需要重新编译binary。


# Linux平台的静态库

## 静态库的生成

     默认就是在编静态库，-c要求只预处理、编译，不链接。
     > gcc -c hello.c
     用ar命令将.o文件归档.a文件。
     > ar -r libhello.a  hello.o

## 静态库的链接

     -static选项是告诉编译器，-L大写的L指明库所在的目录，-l小写的L是在指出需要的动态库，hello是静态库。
     > gcc main.c -static -L .  -lhello  -o  main


# Linux平台的动态库

## 动态库的生成

     在使用GCC编译程序时，只需加上-shared选项,
     > gcc hello.c -fPIC -shared -o libhello.so

## 动态库的链接

     不加-static选项，-l小写的L是在指出需要的动态库。
     > gcc main.c -L . -lhello -o main



# Windows平台的静态库

## 静态库的生成

/MT 使用 LIBCMT.lib 编译以创建多线程可执行文件。  生成静态库lib。

## 静态库的链接

- 1、在使用链接库的代码开头加入，第二行是要调用的链接库里的函数：
```
#pragma comment(lib,"mydll.lib")
extern "C" __declspec(dllimport) int hello(int);
```
- 2、将要调用的链接库的lib放入项目源代码中，然后编译。（编译的时候不需要dll。这里把静态部分lib编译进了exe，但动态库dll还没用。）
- 3、运行之前要把dll放到exe目录下。


# Windows平台的动态库

## 动态库的生成

/MD 使用 MSVCRT.lib 编译以创建多线程 DLL。生成动态库。

## 动态库的链接

- 1、 LoadLibrary（或MFC 的AfxLoadLibrary），装载动态库。
- 2、 GetProcAddress，获取要引入的函数，将符号名或标识号转换为DLL内部地址。
- 3、 FreeLibrary（或MFC的AfxFreeLibrary），释放动态链接库。




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