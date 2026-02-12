---
layout: post
title:  "强化学习算法"
subtitle: ""
categories: [MachineLearning]
---

# 强化学习基本概念

强化学习任务通常用马尔科夫决策过程 MDP 来描述，机器处于环境E，状态空间为X，机器能采取的动作空间为A，状态迁移函数P，奖赏函数R.

强化学习系统的四个要素：策略、收益信号、价值函数、对环境建立的模型。

策略梯度定理


## 强化学习和监督学习的比较
监督学习是从已知数据、已知结果里学习的（从老师学）；而强化学习里，正确的答案是没有人知道的（从经验学习）。

## 强化学习和无监督学习的比较
无监督学习是在无label标注的数据中寻找隐含结构的过程。
强化学习的目标是要最大化收益信号，而不是要找出数据的隐含结构。

## 强化学习方法和进化方法的比较
进化方法是智能体不精确感知环境状态，而只选用一种策略，通过收益最多的策略来产生下一代智能体的策略。
强化学习方法更在意智能体与环境的互动。

## 解决强化学习问题的一般框架 - 有限马尔科夫决策过程

三种基本方法
- 动态规划
- 蒙特卡洛
- 时序差分




# 传统强化学习


## "利用-探索" ( Exploitation/Exploration ) 建模


# 深度强化学习网络
深度强化学习结合了传统的强化学习方法和深度神经网络，使得智能体能够在复杂的环境中学习到更高级别的特征表示，从而做出更高效的决策。

## DQN 算法 Deep Q-network

DDPG


# 基于人类反馈的强化学习 Reinforcement Learning with Human Feedback - RLHF



## PPO（Proximal Policy Optimization）
PPO 的核心思想是：
在策略更新时对变化幅度进行约束，以提高训练稳定性。

## DPO（Direct Preference Optimization）
DPO 的核心思想是：
1. 通过人类反馈（Human Feedback）收集用户偏好数据，例如通过问卷调查、交互日志、A/B 测试等方式获取用户对不同系统输出的偏好信息。
2. 采用二元比较（Pairwise Preference）来训练策略。
3. 直接使用偏好数据优化策略，无需构造显式奖励函数。

## GRPO（Group Relative Policy Optimization）
GRPO的核心思想是：
不训练单独的价值函数，而是通过对比多个策略输出来优化决策。
在训练过程中，每次对同一输入生成多个输出，并计算它们的相对优势。
通过群体平均奖励来指导策略优化。




# 开源项目

https://docs.ray.io/en/master/rllib.html
https://github.com/thu-ml/tianshou



# 参考资料

- gym强化学习工具集
https://gym.openai.com/

- Arcade Learning Environment
https://github.com/mgbellemare/Arcade-Learning-Environment

- 这个baseline库里面有好多注明的强化学习算法实现
https://github.com/openai/baselines 中你所了解的强化学习算法

- 深度强化学习综述 Deep Reinforcement Learning: An Overview
https://arxiv.org/pdf/1810.06339.pdf

- Reinforcement Learning : An introduction
https://item.jd.com/12696004.html

- 斯坦福CS234 Reinforcement Learning
http://web.stanford.edu/class/cs234/index.html

- UCL Course on RL
https://www.davidsilver.uk/teaching/

- 动手学强化学习
https://hrl.boyuai.com/chapter/intro