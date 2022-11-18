import sys
sys.path.insert(0, "../")

from rlcard import make
from rlcard.agents.ac_agent import ACAgent
import torch
from rlcard.agents.nfsp_agent import  NFSPAgent
from rlcard.agents.random_agent import RandomAgent
import numpy as np

import matplotlib.pyplot as plt

env = make("blackjack")
print("Number of actions:", env.num_actions)
print("Number of players:", env.num_players)
print("Shape of state:", env.state_shape)
print("Shape of action:", env.action_shape)

agent = ACAgent(learning_rate_actor=0.001, learning_rate_critic=0.01)

env.set_agents([agent for _ in range(env.num_players)])

from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

x_points = []
y_points = []

for episode in range(50000):
    trajectories, payoffs = env.run(is_training=True)
    trajectories = reorganize(trajectories, payoffs)
    for ts in trajectories[0]:
        agent.feed(ts)

    if episode % 200 == 0:
        ret = tournament(env, 1000)
        print("episode:{}, reward:{}".format(episode, ret))
        x_points.append(episode)
        y_points.append(ret[0])

x_points_np = np.array(x_points)
y_points_np = np.array(y_points)

plt.plot(x_points_np, y_points_np)
plt.show()
