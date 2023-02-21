import sys
sys.path.insert(0, "../")

from rlcard import make
from rlcard.agents.ac_smart_agent import ACSmartAgent
import torch
from rlcard.agents.nfsp_agent import  NFSPAgent
from rlcard.agents.random_agent import RandomAgent
import numpy as np

import matplotlib.pyplot as plt
from copy import deepcopy

env = make("blackjack")
print("Number of actions:", env.num_actions)
print("Number of players:", env.num_players)
print("Shape of state:", env.state_shape)
print("Shape of action:", env.action_shape)

agent_list = []
for i in range(0, env.num_players):
    agent_list.append(ACSmartAgent(learning_rate_actor=0.001, learning_rate_critic=0.01, network_dim=128, num_spaces=env.state_shape[0][0], playerid=i, batch_size=64))

env.set_agents(agent_list)

from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

x_points = []
y0_points = []
y1_points = []

for episode in range(200000):
    trajectories, payoffs = env.run(is_training=True)
    trajectories = reorganize(trajectories, payoffs)


#    print("======00000======")


    for ts in trajectories[0]:
#        (s, a, r, s_, done) = tuple(ts)
#        print("s:", s)
#        print("a:", a)
#        print("r:", r)
#        print("s_:", s_)
#        print("done:", done)
        env.agents[0].feed(tuple(ts))
    
#    print("======11111======")
    
    for ts in trajectories[1]:
#        (s, a, r, s_, done) = tuple(ts)
#        print("s:", s)
#        print("a:", a)
#        print("r:", r)
#        print("s_:", s_)
#        print("done:", done)
        env.agents[1].feed(tuple(ts))

    if episode % 1000 == 0:
        ret = tournament(env, 1000)
        print("episode:{}, reward:{}".format(episode, ret))
        x_points.append(episode)
        y0_points.append(ret[0])
        y1_points.append(ret[1])

x_points_np = np.array(x_points)
y0_points_np = np.array(y0_points)
y1_points_np = np.array(y1_points)

plt.plot(x_points_np, y0_points_np, 'r-', label="player0")
plt.plot(x_points_np, y1_points_np, 'b-', label="player1")
plt.legend()
#plt.savefig("result.png")
plt.show()
