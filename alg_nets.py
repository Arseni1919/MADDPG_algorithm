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

        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            # nn.Tanh()
            nn.Sigmoid(),
        )

        self.n_actions = n_actions
        self.obs_size = obs_size
        self.entropy_term = 0

    def forward(self, state):
        if type(state) is np.ndarray:
            state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        state = state.float()
        value = self.net(state)

        return value


class CriticNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """
    def __init__(self, obs_size: int, n_actions: int, n_agents: int):
        super(CriticNet, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size * n_agents, 64),
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

