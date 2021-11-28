import torch

from alg_GLOBALS import *


class ActorNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """
    def __init__(self, obs_size: int, n_actions: int):
        super(ActorNet, self).__init__()

        self.head_mean = nn.Linear(64, n_actions)
        self.head_log_std = nn.Linear(64, n_actions)

        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Linear(64, n_actions),
            # nn.Tanh()
            # nn.Sigmoid(),
        )

        self.n_actions = n_actions
        self.obs_size = obs_size
        self.entropy_term = 0

    def forward(self, state):
        state = state.float()
        value = self.net(state)
        action_mean = self.head_mean(value)
        action_std = torch.exp(self.head_log_std(value))

        return action_mean, action_std


class CriticNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """
    def __init__(self, obs_size: int, n_actions: int, n_agents: int):
        super(CriticNet, self).__init__()

        self.obs_net = nn.Sequential(
            # nn.Linear(obs_size * n_agents, 64),
            nn.Linear(obs_size, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

        self.n_actions = n_actions
        self.obs_size = obs_size
        self.n_agents = n_agents
        self.entropy_term = 0

    def forward(self, state):
        state = state.float()
        value = self.obs_net(state)
        return value

