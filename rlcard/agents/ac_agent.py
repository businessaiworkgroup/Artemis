#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

# 网络参数初始化，采用均值为 0，方差为 0.1 的高斯分布
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)


# 策略网络
class Actor(nn.Module):
    def __init__(self, num_actions, num_spaces):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(num_spaces, 50), nn.ReLU(),
                                 nn.Linear(50, num_actions))  # 输出为各个动作的概率，维度为2
    def forward(self, s):
        output = self.net(s)
        output = F.softmax(output, dim=-1)  # 概率归一化
        return output


# 价值网络
class Critic(nn.Module):
    def __init__(self, num_spaces):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(num_spaces, 20), nn.ReLU(),
                                 nn.Linear(20, 1))  # 输出值是对当前状态的打分，维度为 1
    def forward(self, s):
        output = self.net(s)
        return output


# A2C 的主体函数
class ACAgent(object):
    def __init__(
        self,
        learning_rate_actor=0.001,
        learning_rate_critic=0.01,
        discount_factor=0.9,
        epsilon=0.9,
        target_replace_iter=100,
        num_actions=2,
        num_spaces=2):

        # 初始化策略网络，价值网络和目标网络。价值网络和目标网络使用同一个网络
        self.actor_net = Actor(num_actions,
                               num_spaces).apply(init_weights)
        self.critic_net = Critic(num_spaces).apply(init_weights)
        self.target_net = Critic(num_spaces).apply(init_weights)
        self.learn_step_counter = 0  # 记录学习步数
        self.optimizer_actor = optim.Adam(self.actor_net.parameters(),
                lr=learning_rate_actor)  # 策略网络优化器
        self.optimizer_critic = \
            optim.Adam(self.critic_net.parameters(),
                       lr=learning_rate_critic)  # 价值网络优化器
        self.criterion_critic = nn.MSELoss()  # 价值网络损失函数
        self.use_raw = False
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.target_replace_iter = target_replace_iter
        self.num_actions = num_actions

    def choose_action(self, s, is_training=True):
        s = torch.unsqueeze(torch.FloatTensor(s), dim=0)  # 增加维度
        if not is_training:
            action_value = self.actor_net(s)
            return torch.max(action_value, dim=1)[1].item()
        if np.random.uniform() < self.epsilon:  # ϵ-greedy 策略对动作进行采取，先固定
            action_value = self.actor_net(s)
            action = torch.max(action_value, dim=1)[1].item()
        else:
            action = np.random.randint(0, self.num_actions)
        return action

    def step(self, state):
        return self.choose_action(state['obs'])

    def eval_step(self, state):
        info = {}
        return (self.choose_action(state['obs'], False), info)

    def feed(self, ts):
        (s, a, r, s_, done) = tuple(ts)
        s = s['obs']
        s_ = s_['obs']

        if self.learn_step_counter % self.target_replace_iter == 0:  # 更新目标网络
            self.target_net.load_state_dict(self.critic_net.state_dict())

        self.learn_step_counter += 1

        s = torch.FloatTensor(s)
        s_ = torch.FloatTensor(s_)

        q_actor = self.actor_net(s)  # 策略网络
        q_critic = self.critic_net(s)  # 价值对当前状态进行打分
        q_next = self.target_net(s_).detach()  # 目标网络对下一个状态进行打分

        q_target = r + self.discount_factor * q_next if not done else torch.tensor([r], dtype=torch.float) # 更新 TD 目标
        td_error = (q_target - q_critic).detach()  # TD 误差

        # 更新价值网络V(s)
        loss_critic = self.criterion_critic(q_critic, q_target)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # 更新策略网络Actor
        dist = Categorical(q_actor)
        log_prob = dist.log_prob(torch.tensor([a], dtype=torch.float))
        actor_loss = -log_prob * td_error
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
