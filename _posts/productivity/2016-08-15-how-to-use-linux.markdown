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


gcc
gcc (GCC) 4.4.7 20120313 (Red Hat 4.4.7-4)


libstdc++
libstdc++.so.6.0.13
 $ strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX


make
GNU Make 3.81


libc
libc-2.12.so


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