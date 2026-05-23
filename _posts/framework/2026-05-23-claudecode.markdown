---
title: "Claude Code生态"
categories: [framework]
layout: post
---

LLM模型本身是大脑；
需要一层harness作为手脚来干活。

# Claude 和 Claude code
Anthropic

plugins ·  skills ·  agents ·  hooks ·  plugin MCP servers ·  plugin LSP servers

plugin
plugin marketplace

harness

  ┌────────────────┬────────────────────────┐
  │      角色      │          类比          │
  ├────────────────┼────────────────────────┤
  │ Claude（模型）   │ 大脑里的思考           │
  ├────────────────┼────────────────────────┤
  │ Harness        │ 身体（手、眼、耳）     │
  ├────────────────┼────────────────────────┤
  │ Hooks          │ 反射动作（不经大脑）   │
  ├────────────────┼────────────────────────┤
  │ MCP servers    │ 外部工具（电话、电脑） │
  └────────────────┴────────────────────────┘


Playwright MCP Bridge
agent-browser

## 一堆抽象的加载词
https://zhuanlan.zhihu.com/p/2028252076333875211

# 新手疑问
* claudecode可以做什么
* skill是自己写的还是ai生成的？
* mcp怎么用？
* 命令、代理、技能、钩子 是什么
* plan是什么

# Claude code的入口

olamma + claude code

ccswitch + claude code

claude-code-router + claude code


# 用ClaudeCode做什么


## 带我了解一个代码项目

## 性能优化
优化模型结构或推理静态图，使其逻辑等价但推理性能更好

# 使用心得
* 我更关注我的目的(原始诉求)，而非实现手段


# claude code 实现原理

# 设计模式
ReAct
plan and execute

三大持久化机制：Rules / Skills / Memory

