import sys
sys.path.insert(0, "../")

from rlcard import make
from rlcard.agents.ac_agent import ACAgent
import torch
from rlcard.agents.nfsp_agent import  NFSPAgent
from rlcard.agents.random_agent import RandomAgent
import numpy as np

env = make("blackjack")
print("Number of actions:", env.num_actions)
print("Number of players:", env.num_players)
print("Shape of state:", env.state_shape)
print("Shape of action:", env.action_shape)

agent = ACAgent()

env.set_agents([agent for _ in range(env.num_players)])

from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

for episode in range(100):
    trajectories, payoffs = env.run(is_training=True)
    trajectories = reorganize(trajectories, payoffs)

    for ts in trajectories[0]:
        agent.feed(ts)


print("-----------eval------------")
print("-----------eval------------")
trans = []
for episode in range(100):
    trajectories, payoffs = env.run(is_training=False)
    trajectories = reorganize(trajectories, payoffs)
    tras = trajectories

    for ts in tras[0]:
        (state, action, reward, next_state, done) = tuple(ts)
#        action = agent.choose_action(state["obs"], False)
#        print(action)
