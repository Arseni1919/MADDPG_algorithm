from alg_constrants_amd_packages import *


class ActorNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """
    def __init__(self, obs_size: int, n_actions: int):
        super(ActorNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, n_actions),
            nn.Tanh()
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
            nn.Linear(obs_size * n_agents + n_actions * n_agents, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + n_actions, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

        self.n_actions = n_actions
        self.obs_size = obs_size
        self.n_agents = n_agents
        self.entropy_term = 0

    def forward(self, state, action):
        if type(state) is np.ndarray:
            state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        state = state.float()
        obs = self.obs_net(state)
        value = self.out_net(torch.cat([obs, action], dim=1))

        return value

