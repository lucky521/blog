---
title: "Linux 远程开发环境部署习惯"
subtitle: ""
categories: [productivity]
layout: post
---

# 个性化配置
## .ssh

## .zshrc .bashrc .bash_profile

## yum.repo.d

## env

## crontab

## 常用的bin 放在~/ludev-bin

## mount 远程磁盘




# 必备软件

## 基本命令
有的新机器甚至连hostname都没有


cmake
cmake version 2.6-patch 4
https://src.fedoraproject.org/repo/pkgs/cmake/cmake-2.6.4.tar.gz/
Installing: /usr/local/bin/cmake


```GCC 属于‌编译工具链‌（含 Binutils），聚焦代码转化。Glibc 属于‌运行时库‌，提供操作系统与应用程序间的桥梁。两者共同构成 Linux 应用开发生态的基础支柱。升级 glibc 的风险整体上比升级 gcc 更大更系统级，不是一个量级。升级 gcc 主要是“编译工具链/运行库版本变更，影响以后编译出来的程序”；升级 glibc 则是直接换掉“几乎所有进程都在用的底层 C 库”，直接影响现有系统二进制和发行版集成。所以通常认为：升级 glibc 更“危险”，尤其是在生产环境/老系统上直接暴力替换。```

gcc
gcc (GCC) 4.4.7 20120313 (Red Hat 4.4.7-4)

g++ --version
g++ (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)

glibc
 $ ldd --version
 $ strings /lib64/libc.so.6 | grep GLIBC_
 https://sourceware.org/glibc/


libc
libc-2.12.so

libstdc++
libstdc++.so.6.0.13
 $ strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
 $ strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX


make
GNU Make 3.81


vim
version 7.4.944 (jd-dev)
version 7.4.1689 (ubuntu16)
youcompleteme need to update Vim 7.4.1578


git
git version 1.7.1


## oh-my-zsh
先安装，然后把自己的.zshrc拷贝过来。

## python
https://www.cnblogs.com/thescholar/p/12167964.html

## java 和 javac


## 高版本gcc
https://blog.csdn.net/qq_39547794/article/details/130080464
```
sudo yum install devtoolset-11-gcc*
scl enable devtoolset-11 bash
g++ --version

```


## bazel

--jobs= 根据编译机器的core核数设定合适

## ruby
安装 rvm、 gem、ruby 挺考验耐心的，版本差异比较大
https://ruby-china.org/wiki/rvm-guide

https://blog.csdn.net/dazhi_100/article/details/38845689

## maven



# 其他环节中关于该机器的配置


# 本地电脑的Remote开发配置
主要体现在本地电脑的 .ssh 中

```
Host bastion
    User lu.dev
    Port 22
    Hostname bastion.xx.com
    IdentitiesOnly yes

Host jdsaa
    User admin
    Port 22
    Hostname 11.49.146.242
    IdentitiesOnly yes
    ProxyCommand ssh bastion -W %h:%p
```