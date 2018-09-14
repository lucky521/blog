---
title: "Python的GIL"
subtitle: "Global Interpreter Lock"
categories: [Language]
layout: post
---
# 什么是Python的GIL？
严格来讲GIL不是Python的。Python作为编程语言是没有GIL这个概念的，有GIL的是某些Python解释器，它们在实现Python语言时对线程的处理方式有所不同。


# 干什么用的？

GIL可以翻译为`全局解释器锁`，用于python解释器对多线程的序列化。在多核系统中，本来多核可以并行多个线程的。GIL就就意味着不能使用多核了，也就是说，即使对于多核系统，同一时间也只有一个拥有GIL锁的python线程在运行。GIL存在于CPython和PyPy解释器里，Jython和IronPython没有GIL。


# 为什么CPython会用GIL？

- 对于单核系统来说，GIL有优化作用，加速多线程程序在单核系统中运行。
- 对线程不安全的C语言库比较友好。
- CPyhon的内存管理是线程不安全的。

# GIL必须取缔吗？
虽然GIL使得一个Python进程中的多线程不可以使用多核。但是Python的多进程`multiprocessing`程序是可以是用多核的。Python主线程fork多个子进程，然后一个核运行一个进程是可以的。


<!--
这里是注释区


{% highlight python %}
print "hello, Lucky!"
{% endhighlight %}

![My image]({{ site.baseurl }}/images/emule.png)

My Github is [here][mygithub].

[mygithub]: https://github.com/lucky521



-->