# Artemis

## Branch `player_interaction_zmy` is the codebase for Players Playing against each other

### 1. Agents implemented
`rlcard.agents.dqn_agent.DQNAgent()`: the agent using DQN policy

`agents.dsac_agent.ACAgent()`: the agent using actor-critic for training

`rlcard.agents.sac_agent.SAC()`: the agent using soft actor-critic for training

### 2. Environment
Based on the original BlackJack Env from [RLcard](https://github.com/datamllab/rlcard), the major change has been made to `rlcard/envs/blackjack.py`, `rlcard/envs/env_player.py`, `rlcard/games/blackjack/game_backup.py` and `rlcard/games/blackjack/dealer.py` for adding the distribution of the seen cards,  and the adversarial interaction between the agents.

`rlcard.envs.blackjack.BlackjackEnv`: add the observation of the other players' hand into the state

`rlcard.envs.env_player.Env()`: the environment with adversarial interaction between the agents

`rlcard.games.balckjack.game_backup.BlackjackGame()`: the game rule with adversarial interaction between the agents

`rlcard.games.balckjack.dealer.BlackjackDealer()`: the dealer with the ability to fold some cards and add into the seen cards for analysing the influence of some known card distribution on the win rates of the players.

`rlcard.utils.utils.tournament(env, num)`: add the win rate of the players

### Train the agents:
```
python main_test.py 
```

### Evaluate the agents:
```
python evaluation.py 
```
