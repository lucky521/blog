---
title: "Javascript学习手册"
categories: [Language]
layout: post
---

JS是解释型语言，无需编译过程。
JS的主要应用是在浏览器上运行，同时它还跨界可以在服务器上运行，在服务器端运行的js也称为node.js。



# 前端JS

## 重要的object
* JavaScript 内置对象：Array, Number, Date...
* Browser对象：
  * Window 表示浏览器中打开的窗口
  * Screen 对象 包含有关客户端显示屏幕的信息
  * Location 对象包含有关当前 URL 的信息
  * History 对象包含用户（在浏览器窗口中）访问过的 URL
  * Storage 对象  sessionStorage （会话存储） 和 localStorage（本地存储）两个存储对象来对网页的数据进行添加、删除、修改、查询操作。localStorage 用于长久保存整个网站的数据，保存的数据没有过期时间，直到手动去除。sessionStorage 用于临时保存同一窗口(或标签页)的数据，在关闭窗口或标签页之后将会删除这些数据。
* HTML DOM对象
  * Document 对象
  * Document 对象是 HTML 文档的根节点。 Document 对象使我们可以从脚本中对 HTML 页面中的所有元素进行访问。Document 对象是 Window 对象的一部分，可通过 window.document 属性对其进行访问。



# 后端JS


```js
var http = require('http');

http.createServer(function (request, response) {

    // 发送 HTTP 头部 
    // HTTP 状态值: 200 : OK
    // 内容类型: text/plain
    response.writeHead(200, {'Content-Type': 'text/plain'});

    // 发送响应数据 "Hello World"
    response.end('Hello World\n');
}).listen(8888);

// 终端打印如下信息
console.log('Server running at http://127.0.0.1:8888/');
```

* require 指令：在 Node.js 中，使用 require 指令来加载和引入模块，引入的模块可以是内置模块，也可以是第三方模块或自定义模块。
* http模块
运行： node xxx.js



# 基础语法


## async/await

在函数前面的 “async” 这个单词表达了一个简单的事情：即这个函数总是返回一个 promise。其他值将自动被包装在一个 resolved 的 promise 中

await 实际上会暂停函数的执行，直到 promise 状态变为 settled，然后以 promise 的结果继续执行。这个行为不会耗费任何 CPU 资源，因为 JavaScript 引擎可以同时处理其他任务：执行其他脚本，处理事件等。



