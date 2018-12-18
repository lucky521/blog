---
title: "Shell中的符号"
subtitle: ""
categories: [Language]
layout: post
---

# 标点符号

## 分号
`;` 分号和换行的作用一样，都是为了分割两条命令。
一行只有一句时，不需要分号，如果一行内有多句，中间就需要分号。if then放在同一行的话也要分号隔开。

## 小括号
`(命令串)` 
重新开一个子shell执行括号内的命令。

## 花括号
`{命令串}` 在当前shell执行命令。
注意：其中的最后一条命令也需要有分号。
第一个命令和左括号之间必须要有一个空格。

## 方括号
方括号起到test逻辑判断的作用。

- `[ ]`：一个方括号，是bash的命令。

    - -x 参数判断$myPath是否存在并且是否具有可执行权限
        [ ! -x "$myPath"]

    - -d 参数判断$myPath是否存在
        [ ! -d "$myPath"]

    - -f 参数判断$myFile是否存在
        [ ! -f "$myFile" ]

- `[[ ]]`：一对方括号是一个方括号的加强版，则是Shell的保留字，里边支持了 `|| &&` 等等这些逻辑运算符号。

## 感叹号

- `!` 当后面跟随的字符不是“空格、换行、回车、=和(”时，做命令替换。
- `!` 当后面是个数字n时，会指向shell 历史命令的第n跳命令。负数的话就是倒数第几条。

## dot点
`.` A dot in that context means to "source" the contents of that file into the current shell.

## 反引号
\` \` 命令替换，将其中的字符串当成shell命令执行



# 特殊符号

## 一个&
command &  让命令在后台运行。

command1 & command2  两个命令同时执行。

## 两个&
command1 && commmand2  command1先执行，只有当command1执行成功才会执行command2，否则command2将不会被执行。


# 特殊变量
- $0：当前脚本的文件名
- $num：num为从1开始的数字，$1是第一个参数，$2是第二个参数，${10}是第十个参数
- $#：传入脚本的参数的个数
- $*：所有的位置参数(作为单个字符串)
- $@：所有的位置参数(每个都作为独立的字符串)。
- $?：当前shell进程中，上一个命令的返回值，如果上一个命令成功执行则$?的值为0，否则为其他非零值，常用做if语句条件
- $$：当前shell进程的pid
- $!：后台运行的最后一个进程的pid
- $-：显示shell使用的当前选项
- $_：之前命令的最后一个参数



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