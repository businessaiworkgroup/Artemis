#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE is:{}".format(DEVICE))

# 网络参数初始化，采用均值为 0，方差为 0.1 的高斯分布
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)


# 策略网络
class Actor(nn.Module):
    def __init__(self, num_actions, num_spaces, dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(num_spaces, dim), nn.ReLU(),
                                 nn.Linear(dim, num_actions))  # 输出为各个动作的概率，维度为num_actions
        if torch.cuda.is_available():
            self.net = self.net.cuda()

    def forward(self, s):
        output = self.net(s)
        output = F.softmax(output, dim=-1)  # 概率归一化
        return output


# 价值网络
class Critic(nn.Module):
    def __init__(self, num_spaces, dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(num_spaces, dim), nn.ReLU(),
                                 nn.Linear(dim, 1))  # 输出值是对当前状态的打分，维度为 1
        if torch.cuda.is_available():
            self.net = self.net.cuda()

    def forward(self, s):
        output = self.net(s)
        return output


# A2C 的主体函数
class ACSmartAgent(object):
    def __init__(
        self,
        learning_rate_actor=0.001,
        learning_rate_critic=0.01,
        discount_factor=0.9,
        epsilon=0.9,
        target_replace_iter=100,
        num_actions=2,
        num_spaces=2,
        network_dim=128,
        batch_size=8,
        playerid=-1):

        # actor critic network
        self.actor_net = Actor(num_actions,
                               num_spaces, network_dim).apply(init_weights)
        self.critic_net = Critic(num_spaces, network_dim).apply(init_weights)
        self.target_net = Critic(num_spaces, network_dim).apply(init_weights)

        # optimizer
        self.optimizer_actor = optim.Adam(self.actor_net.parameters(),
                lr=learning_rate_actor)
        self.optimizer_critic = \
            optim.Adam(self.critic_net.parameters(),
                       lr=learning_rate_critic)

        # loss function
        self.criterion_critic = nn.CrossEntropyLoss()

        self.learn_step_counter = 0
        self.use_raw = False
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.target_replace_iter = target_replace_iter
        self.num_actions = num_actions
        self.playerid = playerid
        # feed memory
        self.feed_memory = []
        self.batch_size = batch_size

    def _choose_action(self, s, is_training=True):
        s = torch.tensor(s, device=DEVICE)
        s = s.float()
        s = torch.unsqueeze(s, dim=0)  # 增加维度
        if not is_training:
            action_value = self.actor_net(s)
            return torch.max(action_value, dim=1)[1].item()
        if np.random.uniform() < self.epsilon:  # ϵ-greedy 策略对动作进行采取，先固定
            action_value = self.actor_net(s)
            action = torch.max(action_value, dim=1)[1].item()
        else:
            action = np.random.randint(0, self.num_actions)
        return action

    # return the inferenced total point of the other player
    def guess_opposite_hidden_card(self):
        return 0

    def step(self, state):
        # print("id:{},state:{}".format(self.playerid, state))
        return self._choose_action(state['obs'])

    def eval_step(self, state):
        info = {}
        return (self._choose_action(state['obs'], False), info)

    def feed(self, ts):
        # update target network
        if self.learn_step_counter % self.target_replace_iter == 0:  # 更新目标网络
            self.target_net.load_state_dict(self.critic_net.state_dict())
        # feed memory
        if self.learn_step_counter % self.batch_size != 0 or self.learn_step_counter == 0:
            self.feed_memory.append(ts)
            self.learn_step_counter += 1
            return

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        for item in self.feed_memory:
            (s, a, r, s_, done) = item
            state_batch.append(s['obs'])
            action_batch.append(a)
            reward_batch.append(r)
            next_state_batch.append(s_['obs'])
            done_batch.append(done)
        self.feed_memory.clear()

        # convert to numpy to speed up to tensor
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch)

        # convert to tensor
        state_batch = torch.from_numpy(state_batch).float().to(DEVICE)
        action_batch = torch.from_numpy(action_batch).long().to(DEVICE)
        reward_batch = torch.from_numpy(reward_batch).long().to(DEVICE)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(DEVICE)
        done_batch = torch.from_numpy(done_batch).long().to(DEVICE)

        # run network
        q_actor = self.actor_net(state_batch)
        q_critic = self.critic_net(state_batch).squeeze()
        q_next = self.target_net(next_state_batch).squeeze()
        q_target = torch.where(
                done_batch == 0, 
                reward_batch + torch.mul(q_next, self.discount_factor).squeeze(), 
                reward_batch)
        td_error = (q_target - q_critic).detach()

        # update value
        loss_critic = self.criterion_critic(q_critic, q_target)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # update actor
        dist = Categorical(q_actor)
        log_prob = dist.log_prob(action_batch)
        actor_loss = torch.mean(-log_prob * td_error)
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # feed memory
        self.feed_memory.append(ts)
        self.learn_step_counter += 1
