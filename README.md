# Artemis

## Branch `opta2c_zmy` is the codebase for Standard training of the agents
### 1. Agents implemented
`rlcard.agents.dqn_agent.DQNAgent()`: the agent using DQN policy

`agents.dsac_agent.ACAgent()`: the agent using actor-critic for training

`rlcard.agents.sac_agent.SAC()`: the agent using soft actor-critic for training

### 2. Environment
Based on the original BlackJack Env from [RLcard](https://github.com/datamllab/rlcard), the major change has been made to `rlcard/envs/blackjack.py`, `rlcard/envs/env_player.py`, `rlcard/games/blackjack/game_backup.py` and `rlcard/games/blackjack/dealer.py` for adding the distribution of the seen cards and the adversarial interaction between the agents.

`rlcard.envs.env_backup.Env()`: the standard environment

`rlcard.envs.env_ac.Env()`: the standard environment for training the SAC agents

`rlcard.envs.env_hand_ac.Env()`: the environment with player action of asking for card in prioirty for training the SAC agents

`rlcard.envs.env.Env()`: the environment with player action of asking for card in prioirty for training the DQN agents

`rlcard.envs.env.blackjack.BlackjackEnv()`: changed to fit the state action space of the agents with different game rules and card recordinf stragedy.

`rlcard.games.balckjack.game_backup.BlackjackGame()`: the standard game rule with player action of stand and hit

`rlcard.games.balckjack.game.BlackjackGame()`: the game rule with player action of hit for card in prioirty, hit and stand

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
