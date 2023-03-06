from rlcard import make
from rlcard.agents.dqn_agent import DQNAgent
import torch
from rlcard.agents.nfsp_agent import  NFSPAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.dsac_agent import  ACAgent

env = make("blackjack")
print("Number of actions:", env.num_actions)
print("Number of players:", env.num_players)
print("Shape of state:", env.state_shape)
print("Shape of action:", env.action_shape)

# agent = DQNAgent(
#     num_actions=env.num_actions,
#     state_shape=env.state_shape[0],
#     mlp_layers=[64,64],
# )

agent = ACAgent(num_actions=env.num_actions,
    state_shape=env.state_shape[0])

# agent = NFSPAgent(
#     num_actions=env.num_actions,
#     state_shape=env.state_shape[0],
#     hidden_layers_sizes=[64,64],
#     q_mlp_layers=[64,64],
# )

# agent = RandomAgent(env.num_actions)
share_policy = True
if share_policy:
    env.set_agents([agent])
else:
    env.set_agents([agent for _ in range(env.num_players)])

from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

best_score = -1
# agent.q_estimator.supervised_actionnet.load_state_dict(torch.load('C:/Users/13862/Desktop/RA_BJ/model_cifar.pt'))
# agent.q_estimator.qnet.load_state_dict(torch.load('C:/Users/13862\Desktop/rlcard-master/experiments/leduc_holdem_dqn_result/supervised_rl/Batch1024_12state_exp0.1_0.05(2)/model_cifar.pt'))
# agent.q_estimator.qnet.load_state_dict(torch.load('experiments/leduc_holdem_dqn_result/fd_fold10_3232/model_cifar.pt'))
# with Logger("experiments/leduc_holdem_dqn_result/sn_4_1(2)/") as logger:

agent.actor_net.load_state_dict(torch.load('C:/Users/13862/Desktop/rlcard-master/Actor_model_cifar.pt'))
agent.critic_net.load_state_dict(torch.load('C:/Users/13862/Desktop/rlcard-master/Critic_model_cifar.pt'))
for episode in range(20): # 1000->1000

    # Generate data from the environment

    trajectories, payoffs = env.run(is_training=False)
    # trajectories, payoffs = env.run(is_training=False)

    # Reorganaize the data to be state, action, reward, next_state, done
    trajectories = reorganize(trajectories, payoffs)
    score = tournament(env, 50000, )[0]
    print(score)
        # Feed transitions into agent memory, and train the agent
        # for ts in trajectories[0]:
        #     agent.feed(ts)

        # Evaluate the performance.
        # if episode % 50 == 0:
        #     score = tournament(env, 10000, )[0]
        #
        #     logger.log_performance(
        #        str( episode)+" " + str(env.timestep),
        #     score
        #     )

    # Get the paths
    # csv_path, fig_path = logger.csv_path, logger.fig_path

