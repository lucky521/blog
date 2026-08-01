---
title: "Claude Code &C odex生态"
categories: [framework]
layout: post
---

LLM模型本身是大脑；
需要一层harness作为手脚来干活。

# 新手概念
Anthropic

plugins ·  skills ·  agents ·  hooks ·  plugin MCP servers ·  plugin LSP servers


hook:
* Fact-Forcing Gate

task

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


## mcp的部署形态
stdio (本地)
Streamable HTTP (远程)


# 新手疑问
* claudecode可以做什么
* skill是自己写的还是ai生成的？
* mcp怎么用？
* 命令、代理、技能、钩子 是什么
* plan是什么


## claude 使用遇到的问题
禁止claude code自动更新   export DISABLE_AUTOUPDATER=1

安装特定版本的claude code 版本： npm install -g @anthropic-ai/claude-code@2.1.90

unset MAX_THINKING_TOKENS
claude --thinking disabled


# Claude Code


## 一堆抽象的加载词
https://zhuanlan.zhihu.com/p/2028252076333875211

## 高频快捷键

Ctrl + Enter：在当前输入框中创建新的一行，不会提交输入


## Claude code的入口

* olamma + claude code
* ccswitch + claude code
* claude-code-router + claude code
 * ccr ui 配置 http://127.0.0.1:3456/ui/
 * ~/.claude-code-router/config.json

### 重要的插件

https://github.com/thedotmack/claude-mem
https://github.com/affaan-m/ECC
superpowers

### 查看、操作web浏览器

浏览器工具：
Chrome DevTools Protocol 原生客户端
Playwright MCP Bridge
agent-browser



## 用ClaudeCode做什么


### 带我了解一个代码项目

### 性能优化
优化模型结构或推理静态图，使其逻辑等价但推理性能更好



## 使用心得

### 怎么更好的管理会话(其实是管理上下文记忆)？
 /save-session + /resume-session：显式保存/恢复（适合工作日结束）

 用 Plan / Task

### subagent怎么用？

### 什么时候需要自己写skill？
⏺ 简短答：当一段"做事方法"满足下面三条同时成立时，才值得写成 skill —— 否则用更轻量的载体。

  写 skill 的判断标准：
  1. 会反复触发：未来还会在多次会话里碰到同类任务（不是一次性脚本）。
  2. 有结构化流程：包含若干步骤、检查清单、或需要选择路径的判断逻辑 —— 简单的一句规则不值当。
  3. 跨项目可复用：不是某个具体仓库的 fact（那种放 CLAUDE.md），也不是你个人偏好（那种放 memory）。

以上3点是ai总结的，为了强调“自己写”三个字，我再追加一点，就是这个skill涉及到的步骤太内部了（比如企业内部的系统），不是网络上公开的；

### 怎么写skill？ 写到哪里？

/.claude/skills/your-skill-name/


```
  ---
  name: your-skill-name
  description: Use when [具体触发条件 / 症状]
  ---

  # Skill 名

  ## Overview
  一两句话讲清楚是什么、核心原则。

  ## When to Use
  - 适用场景（症状/错误/关键字）
  - 不适用的情况

  ## Core Pattern / Quick Reference
  表格或 before/after 代码对比，方便扫读。

  ## Implementation
  一个**优秀完整**的示例（别堆 5 种语言版本）。

  ## Common Mistakes
  踩过的坑 + 修法。
```



### 更关注我的目的(原始诉求)，而非实现手段

### AI是一点就透的聪明人，不用细教


## claude code 实现原理

### 设计模式
ReAct
plan and execute
Tool-Use Loop

三大持久化机制：Rules / Skills / Memory


ANTHROPIC_BASE_URL
ANTHROPIC_AUTH_TOKEN
ANTHROPIC_API_KEY


### 五级递进式上下文压缩


### 低成本多Agent并行架构



# Codex

中转方式： 基于ccswitch中转
打开方式：终端进入项目路径下，然后 codex 或者 codex app

核心配置：~/.codex/config.toml


# hermes

All your files are in ~/.hermes/:

   Settings:  ~/.hermes/config.yaml
   API Keys:  ~/.hermes/.env
   Data:      ~/.hermes/cron/, sessions/, logs/


# oopenhands


# Troubleshoot

Anthropic 的api协议和 openai的 api协议有什么差异

claude模型调用报错：
* ValidationException: \\\"***.***.enabled\\\" is not supported for this model. Use \\\"***.***.adaptive\\\" and \\\"output_config.effort\\\" to control thinking behavior
* API Error: 400 参数解析失败


codex模型调用报错：
* Model metadata for `GPT-5.5-xxx` not found. Defaulting to fallback metadata; this can degrade performance and cause issues.
* stream disconnected before completion: stream closed before response.completed


下载的plugin怎么丢了？