from rlcard import make
from rlcard.agents.dqn_agent import DQNAgent
import torch
from rlcard.agents.nfsp_agent import  NFSPAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.sac_agent import  SAC

env = make("blackjack")
print("Number of actions:", env.num_actions)
print("Number of players:", env.num_players)
print("Shape of state:", env.state_shape)
print("Shape of action:", env.action_shape)

# agent = DQNAgent(
#     num_actions=env.num_actions,
#     state_shape=env.state_shape[0],
#     mlp_layers=[64, 64],
# )

# agent = NFSPAgent(
#     num_actions=env.num_actions,
#     state_shape=env.state_shape[0],
#     hidden_layers_sizes=[64,64],
#     q_mlp_layers=[64,64],
# )

agent = SAC(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0]
)

# agent = RandomAgent(env.num_actions)

env.set_agents([agent for _ in range(env.num_players)])

from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

best_score = -1
agent.load_checkpoint('checkpoints/sac_checkpoint_blackjack_')
# agent.load_checkpoint('experiments/leduc_holdem_dqn_result/normalised_obs/Soft_AC/full_dis_10_fold/sac_checkpoint_blackjack_')
# with Logger("experiments/leduc_holdem_dqn_result/sn_4_1(2)/") as logger:
for episode in range(20): # 1000->1000

    # Generate data from the environment
    # agent.q_estimator.qnet.load_state_dict(torch.load('experiments/leduc_holdem_dqn_result/multiagents_sharedpolicy/Soft_AC/sac_checkpoint_blackjack_'))
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

