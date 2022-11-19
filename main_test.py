from rlcard import make
from rlcard.agents.dqn_agent import DQNAgent
import torch
from rlcard.agents.nfsp_agent import  NFSPAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.sac_agent import  SAC
from rlcard.agents.dsac_agent import  ACAgent

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
#
agent = ACAgent(num_actions=env.num_actions,
    state_shape=env.state_shape[0])

# agent = NFSPAgent(
#     num_actions=env.num_actions,
#     state_shape=env.state_shape[0],
#     hidden_layers_sizes=[64,64],
#     q_mlp_layers=[64,64],
# )

# agent = NFSPAgent(num_actions=env.num_actions,
#                 state_shape=env.state_shape[0],
#                 hidden_layers_sizes=[10,10],
#                 q_mlp_layers=[10,10])

# agent = SAC(
#     num_actions=env.num_actions,
#     state_shape=env.state_shape[0]
# )
a2c = True
sac = False
NFSP = False
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
is_training = True

# agent.q_estimator.qnet.supervised_actionnet.load_state_dict(
#             torch.load('Supervised_Q_model_cifar.pt'))
# agent.q_estimator.qnet.load_state_dict(
#             torch.load('Supervised_Q_model_cifar.pt'))

# if a2c:
#     # agent.actor_net.load_state_dict(torch.load('Supervised_Actor.pt'))
#     # pretrained_dict = torch.load('Supervised_Actor.pt')
#     agent.actor_net.load_state_dict(torch.load('BN_Actor.pt'))
#     pretrained_dict = torch.load('BN_Actor.pt')
#     model_dict = agent.critic_net.state_dict(pretrained_dict)
#     print(len(pretrained_dict.items()))
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     print(len(pretrained_dict.items()))
#     model_dict.update(pretrained_dict)
#     agent.critic_net.load_state_dict(pretrained_dict)


with Logger("experiments/leduc_holdem_dqn_result/") as logger:
    for episode in range(50000): # 1000->1000

        # Generate data from the environment

        trajectories, payoffs = env.run(is_training=is_training)

        # trajectories, payoffs = env.run(is_training=False)
        # print(trajectories)
        # Reorganaize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)

        # Feed transitions into agent memory, and train the agent
        if is_training:
            for ts in trajectories[0]:
                agent.feed(ts)

        # Evaluate the performance.
        if episode % 50 == 0:
            score = tournament(env, 15000, )[0]
            # print(agent.total_t)
            if score > best_score and is_training:
                if sac:
                    agent.save_checkpoint('blackjack')
                elif a2c:
                    torch.save(agent.actor_net.state_dict(), 'Actor_model_cifar.pt')
                    torch.save(agent.critic_net.state_dict(), 'Critic_model_cifar.pt')
                elif NFSP:
                    torch.save(agent._rl_agent.q_estimator.qnet.state_dict(), 'model_cifar.pt')
                else:
                    torch.save(agent.q_estimator.qnet.state_dict(), 'model_cifar.pt')
                print('Model Saved Score from', best_score, 'to', score)
                best_score = score
            logger.log_performance(
               str( episode)+" " + str(env.timestep),
            score
            )

    # Get the paths
    csv_path, fig_path = logger.csv_path, logger.fig_path

