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
        self.transit_one_batch = []

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

    def _compute_returns(self, rewards, masks, gamma):
        R = 0
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def feed(self, transit):
        (s, a, r, s_, done) = tuple(transit)
        self.transit_one_batch.append(transit)
        if not done:
            return
        log_probs = []
        critic_vals = []
        masks = []
        rewards = []

        for ts in self.transit_one_batch:
            (s, a, r, s_, done) = tuple(ts)
            s = s['obs']
            s = torch.FloatTensor(s)
            # actor predict
            act_prob = self.actor_net(s)
            act_prob = Categorical(act_prob)
            log_prob = act_prob.log_prob(torch.tensor([a], dtype=torch.float)).unsqueeze(0)

            critic_val = self.critic_net(s)

            log_probs.append(log_prob)
            critic_vals.append(critic_val)
            masks.append(torch.tensor([1-done], dtype=torch.float))
            rewards.append(torch.tensor([r], dtype=torch.float))

        # target value
        returns = self._compute_returns(rewards, masks, self.discount_factor)
        log_probs = torch.cat(log_probs)
        critic_vals = torch.cat(critic_vals)
        returns = torch.cat(returns)

        td_error = returns - critic_vals

        actor_loss = -(log_probs * td_error.detach()).mean()
        critic_loss = td_error.pow(2).mean()

        # update parameters
        self.optimizer_critic.zero_grad()
        self.optimizer_actor.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.optimizer_critic.step()
        self.optimizer_actor.step()
        # finish current trajectory
        self.transit_one_batch.clear()
