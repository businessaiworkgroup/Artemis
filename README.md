# Artemis

## Branch `opta2c_zmy` is the codebase for Standard training of the agents
### 1. Agents implemented
`rlcard.agents.dqn_agent.DQNAgent()`: the agent using DQN policy
`agents.dsac_agent.ACAgent()`: the agent using actor-critic for training
`rlcard.agents.sac_agent.SAC()`: the agent using soft actor-critic for training

### 2. Environment
Based on the original BlackJack Env from [RLcard](https://github.com/datamllab/rlcard)


### Train the agents:
```
python main_test.py 
```

### Evaluate the agents:
```
python evaluation.py 
```
