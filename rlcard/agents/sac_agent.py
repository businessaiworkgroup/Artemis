import random
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from rlcard.agents.utils import  soft_update, hard_update
# from utils import soft_update, hard_update
from rlcard.agents.model import GaussianPolicy, QNetwork, DeterministicPolicy
from collections import namedtuple
from copy import deepcopy

from rlcard.utils.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done'])



class SAC(object):
    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=0.0, # 0 -> 0.8
                 epsilon_end=0,
                 epsilon_decay_steps=10000, # 20000 -> 5000
                 batch_size=32,
                 num_actions=2,
                 state_shape=None,
                 train_every=1, # 1->2
                 learning_rate=0.00005,
                 device=None):

                    # (self, num_inputs, action_space, args):

        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.train_every = train_every
        self.start_steps = 10000

        self.gamma = discount_factor
        self.tau = 0.005
        self.alpha = 0.2 # 0.2 ->0.15
        mlp_layer = 128 # 256 -> 128

        self.train_t=0
        self.total_t = 0

        self.policy_type = "Gaussian"
        self.target_update_interval = train_every*100
        self.automatic_entropy_tuning = False

        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        self.num_actions = num_actions
        num_inputs = state_shape[0]

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.critic = QNetwork(num_inputs, num_actions, mlp_layer).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)

        self.critic_target = QNetwork(num_inputs, num_actions, mlp_layer).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.memory = Memory(replay_memory_size, batch_size)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(num_actions.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)

            self.policy = GaussianPolicy(num_inputs, num_actions, mlp_layer, num_actions).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, num_actions, mlp_layer, num_actions).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

    def step(self, state):
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        # print(epsilon,self.num_actions)
        # epsilon = 0
        random_decision = random.random()
        if random_decision < epsilon:

            action = random.choices([i for i in range(self.num_actions)])[0]
            action_probabililty = np.array(
                [1 / self.num_actions for _ in range(self.num_actions)])
            # print('yes',action)
            return  action,  action_probabililty
        else:
            state = torch.FloatTensor(state['obs']).to(self.device).unsqueeze(0)

            # action_probabilities, _, action = self.policy.sample(state)
            # # print(action_probabilities.detach().cpu().numpy()[0])
            # return action.detach().cpu().numpy()[0][0], action_probabilities.detach().cpu().numpy()[0]
            action, _, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()[0].argmax(), action.detach().cpu().numpy()[0]

    def eval_step(self, state):
        state = torch.FloatTensor(state['obs']).to(self.device).unsqueeze(0)
        # action_probabilities, _, action = self.policy.sample(state)
        # return action.detach().cpu().numpy()[0][0], action_probabilities.detach().cpu().numpy()[0]
        _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0].argmax(), action.detach().cpu().numpy()[0]

    def feed(self, ts):
        # Sample a batch from memory
        state, action, reward, next_state, done = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], list(next_state['legal_actions'].keys()),done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp >= 0 and tmp % self.train_every == 0:
            state_batch, action_batch, reward_batch, next_state_batch, legal_actions_batch, mask_batch = self.memory.sample()
            #     memory.sample(batch_size=batch_size)
            #
            # (state, action, reward, next_state, done) = tuple(memory)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                # print('qf1_next_target',qf1_next_target)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi

                # min_qf_next_target = (next_state_action * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi )).sum(dim=1, keepdim=True)
                # print(min_qf_next_target)

                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            # print(next_state_log_pi)
            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

            # qf1 = qf1.detach().mean().item() ### added
            # qf2 = qf2.detach().mean().item() ### added
            # print(qf1, next_q_value)
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            pi, log_pi, _ = self.policy.sample(state_batch)
            # with torch.no_grad():
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            #### added _________________
            # entropies = -torch.sum(
            #     pi * log_pi, dim=1, keepdim=True)
            # q = torch.sum(min_qf_pi * pi, dim=1, keepdim=True)
            # policy_loss = ( (- q - self.alpha * entropies)).mean()
            ####________________________

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone() # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

            print('\rINFO - Step {}, qf1-loss: {}, qf2-loss: {}'.format(self.total_t, qf1_loss.item(), qf2_loss.item()),
                  end='')
            if self.train_t % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)
                print("\nINFO - Copied model parameters to target network.")
            self.train_t += 1

            return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

    def feed_memory(self, state, action, reward, next_state, legal_actions, done):
        ''' Feed transition to memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        self.memory.save(state, action, reward, next_state, legal_actions, done)


class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, legal_actions, done):
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
        transition = Transition(state, action, reward, next_state, legal_actions, done)
        self.memory.append(transition)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        # print(self.memory)
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))
