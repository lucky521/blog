---
title: "教你熟悉 Django"
subtitle: "Django 使用手册"
categories: [design]
layout: post
---
Django是一款流行的Web框架，完全使用Python打造。内置功能丰富，直接提供的功能有SS：

- 账户权限认证
- 日志系统
- 缓存系统
- 分页功能
- 序列化工具
- 测试工具

# MTV 架构

- M stands for “Model,” the data access layer. This layer contains anything and everything about the data: how to access it, how to validate it, which behaviors it has, and the relationships between the data.
- T stands for “Template,” the presentation layer. This layer contains presentation-related decisions: how something should be displayed on a Web page or other type of document.
- V stands for “View,” the business logic layer. This layer contains the logic that access the model and defers to the appropriate template(s). You can think of it as the bridge between models and templates.




# 官方文档

[docs.djangoproject](https://docs.djangoproject.com/en/1.9/)


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