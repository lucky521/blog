---
title: "教你熟悉 Django"
subtitle: "Django 使用手册"
categories: [framework]
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

- Model 代表“模型”，数据访问层。这一层包含一切有关数据的东西：如何访问数据，如何对其进行验证，哪些行为和数据有关系。
- Template 代表“模板”，表示层。这个层包含展示相关的东西：如何东西应的网页或其他类型的文档上显示。
- View 代表“视图”，业务逻辑层。虽然它名字叫做视图，但不是负责界面展示的。这个层访问模型Model并套取适当的模板Template。可以把它看作模型与模板之间的桥梁。

# 文件结构

- manage.py
- `<project_name>`/settings.py
- `<project_name>`/urls.py
- `<project_name>`/wsgi.py
- `<apps-folder>`/views.py
- `STATICFILES_DIRS/*.[js|css]`
- `TEMPLATE_DIRS/*.html`


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