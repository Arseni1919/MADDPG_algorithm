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
        )

        self.mean_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, n_actions),
        )

        self.std_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, n_actions),
            nn.Softplus()
        )

        self.n_actions = n_actions
        self.obs_size = obs_size
        self.entropy_term = 0

    def forward(self, state):
        if type(state) is np.ndarray:
            state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        state = state.float()
        value = self.net(state)
        value = value.float()
        mean = self.mean_head(value)
        std = self.std_head(value)
        return mean, std

    @torch.no_grad()
    def get_action(self, state):
        mean, std = self(state)
        mean = torch.squeeze(mean)
        std = torch.squeeze(std)
        normal_dist = Normal(loc=mean, scale=std)
        action = torch.tanh(normal_dist.sample())
        action = action.float().detach()
        # log_policy_a_s = normal_dist.log_prob(action) - torch.sum(torch.log(1 - action.pow(2)))
        return action


class CriticNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """
    def __init__(self, obs_size: int, n_actions: int):
        super(CriticNet, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + n_actions, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

        self.n_actions = n_actions
        self.obs_size = obs_size
        self.entropy_term = 0

    def forward(self, state, action):
        if type(state) is np.ndarray:
            state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        state = state.float()
        obs = self.obs_net(state)
        value = self.out_net(torch.cat([obs, action], dim=1))
        value = value.float()

        return value

