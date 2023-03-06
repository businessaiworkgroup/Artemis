import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math, random, copy

# actor learining小，critic大
LR_ACTOR = 5e-4    # 策略网络的学习率
LR_CRITIC = 1e-3   # 价值网络的学习率
GAMMA = 0.9         # 奖励的折扣因子
EPSILON = 1      # ϵ-greedy 策略的概率
TARGET_REPLACE_ITER = 1000                 # 目标网络更新的频率
N_ACTIONS = 2 #env.action_space.n            # 动作数
N_SPACES = 12 #env.observation_space.shape[0] # 状态数量

# 网络参数初始化，采用均值为 0，方差为 0.1 的高斯分布
def init_weights(m) :
    if isinstance(m, nn.Linear) :
        nn.init.normal_(m.weight, mean = 0, std = 0.1)

# 策略网络
# class Actor(nn.Module) :
#     def __init__(self):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(N_SPACES, 64)
#         #         # self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(64, N_ACTIONS)
#         self.activation = nn.Softmax(dim=-1)
#         self.relu = nn.ReLU()
#         # self.net = nn.Sequential(
#         #     nn.Linear(N_SPACES, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, N_ACTIONS) # 输出为各个动作的概率，维度为 3
#         # )
#
#     def forward(self, s):
#         out = self.relu(self.fc1(s))
#         # out = F.relu(self.fc2(out))
#         output = self.activation(self.fc3(out))
#         # output = self.net(s)
#         # output = F.softmax(output, dim = -1) # 概率归一化
#         return output
#
# # 价值网络
# class Critic(nn.Module) :
#     def __init__(self):
#         super(Critic, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(N_SPACES, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1) # 输出值是对当前状态的打分，维度为 1
#         )
#
#     def forward(self, s):
#
#         output = self.net(s)
#         return output


class Actor(nn.Module) :
    def __init__(self,input_size,hidden_size,action_size):
        super(Actor, self).__init__()
        print(input_size)
        self.normalisation = nn.BatchNorm1d(np.prod(input_size))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.activation = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(x.size())
        x = self.flatten(x)
        x = self.normalisation(x)
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.activation(self.fc3(out))
        return out

# 价值网络
class Critic(nn.Module) :

    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.normalisation = nn.BatchNorm1d(np.prod(input_size))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.flatten(x)
        x = self.normalisation(x)
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# A2C 的主体函数
class ACAgent:
    def __init__(self, state_shape=None, mlp_layers=[64, 64], num_actions=2, batch_size = 32):
        # 初始化策略网络，价值网络和目标网络。价值网络和目标网络使用同一个网络
        self.mlp_layers = mlp_layers
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.actor_net, self.critic_net, self.target_net = Actor().apply(init_weights), Critic().apply(
        #     init_weights), Critic().apply(init_weights)
        self.actor_net = Actor(state_shape[0], self.mlp_layers[0], num_actions).apply(init_weights).to(device=self.device)
        self.critic_net = Critic(state_shape[0], self.mlp_layers[0], 1).apply(init_weights).to(device=self.device)
        self.target_net = Critic(state_shape[0], self.mlp_layers[0], 1).apply(init_weights).to(device=self.device)
        self.actor_net.eval()
        self.critic_net.eval()
        self.target_net.eval()
        self.learn_step_counter = 0 # 学习步数
        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr = LR_ACTOR)    # 策略网络优化器
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr = LR_CRITIC) # 价值网络优化器
        self.criterion_critic = nn.MSELoss() # 价值网络损失函数
        self.use_raw = False
        self.EPSILON = EPSILON
        self.EPSILON_STEP = 0.0001
        self.EPSILON_END = 1
        self.memory = Memory(batch_size)
        self.batch_size = batch_size



    def choose_action(self, s, is_training=True):
        s = torch.unsqueeze(torch.FloatTensor(s), dim = 0).to(device=self.device) # 增加维度
        if not is_training:
            action_value = self.actor_net(s)
            return torch.max(action_value, dim = 1)[1].item(), action_value
        if np.random.uniform() < self.EPSILON :                 # ϵ-greedy 策略对动作进行采取，先固定
            action_value = self.actor_net(s)
            action_probabililty = action_value
            action = torch.max(action_value, dim = 1)[1].item()
            # print("action_value training", action_value, action)
        else :
            action_probabililty = np.array(
                [1 / N_ACTIONS for _ in range(N_ACTIONS)])
            action = np.random.randint(0, N_ACTIONS)
        # return action, action_probabililty
        return action

    def step(self, state):
        return self.choose_action(state['obs'])

    def eval_step(self, state):
        state = torch.FloatTensor(state['obs']).unsqueeze(0).to(device=self.device)
        action_value = self.actor_net(state)
        info = {}
        info['values'] = action_value
        # return torch.max(action_value, dim=1)[1].item(), action_value
        return torch.max(action_value, dim=1)[1].item(), info

        # info = {}
        # return self.choose_action(state['obs'], False), info

    def feed(self, ts):

        (s, a, r, s_, done) = tuple(ts)

        s = s['obs']
        s_ = s_['obs']
        self.memory.save(s, a, r, s_, done)
        # a = a.argmax()
        self.learn_step_counter += 1

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0 :          # 更新目标网络
            self.target_net.load_state_dict(self.critic_net.state_dict())

        if self.learn_step_counter % self.batch_size == 0:
            # print('self.learn_step_counter',self.learn_step_counter)
            batch = self.memory.pop_batch()
            s, a, r, s_, done = batch
            s = torch.FloatTensor(s).to(self.device)
            s_ = torch.FloatTensor(s_).to(self.device)
            a = torch.FloatTensor(a).to(self.device).long()
            r = torch.FloatTensor(r).to(self.device).unsqueeze(1)
            done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
            # print(s, a, r, s_, done)
            # s = torch.FloatTensor(s)
            # s_ = torch.FloatTensor(s_)
            self.actor_net.train()
            self.critic_net.train()
            self.target_net.train()

            q_actor = self.actor_net(s)  # 策略网络
            q_critic = self.critic_net(s)  # 价值对当前状态进行打分
            q_next = self.target_net(s_)  # 目标网络对下一个状态进行打分
            # print(s)
            # print('next',s_)
            # print(done)
            # print('(1-done)',(1-done))

            q_target = r + (1 - done) * GAMMA * q_next  # 更新 TD 目标
            # q_target = r + GAMMA * q_next  # 更新 TD 目标
            # 更新价值网络V(s)
            loss_critic = (q_target - q_critic).pow(2).mean()
            # loss_critic = self.criterion_critic(q_critic, q_target)
            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

            q_critic = self.critic_net(s)  # 价值对当前状态进行打分
            q_next = self.target_net(s_)  # 目标网络对下一个状态进行打分

            q_target = r + (1 - done) * GAMMA * q_next  # 更新 TD 目标
            # q_target = r +  GAMMA * q_next  # 更新 TD 目标
            # td_error = torch.abs(q_target - q_critic)  # TD 误差
            td_error = q_target - q_critic  # TD 误差

            # 更新策略网络Actor
            # log_q_actor = torch.log(q_actor)
            # actor_loss = (log_q_actor[a] * td_error).mean()

            dist = torch.distributions.Categorical(q_actor)
            log_probs = dist.log_prob(a).view(-1, 1)
            actor_loss = -(log_probs * td_error).mean()


            # print(actor_loss)
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.actor_net.eval()
            self.critic_net.eval()
            self.target_net.eval()

            if self.EPSILON < self.EPSILON_END:
                self.EPSILON += self.EPSILON_STEP





class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, done):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = [state, action, reward, next_state, done]
        self.memory.append(transition)

    def pop_batch(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = self.memory
        # print(samples)
        self.memory = []
        return map(np.array, zip(*samples))