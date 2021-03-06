import gym.spaces
import pettingzoo
import torch

from alg_GLOBALS import *


class MultiAgentParallelEnvWrapper:
    """
    The main duties of this wrapper:

    1. Receive inputs and transform outputs as tensors
    2. Normalize states of observations

    """

    def __init__(self, parallel_env):
        if not isinstance(parallel_env, pettingzoo.ParallelEnv):
            raise RuntimeError(f'~[ERROR]: Not a parallel env!')

        self.parallel_env = parallel_env
        observations = self.parallel_env.reset()

        self.agents = self.parallel_env.agents
        self.num_agents = self.parallel_env.num_agents
        self.max_num_agents = self.parallel_env.max_num_agents
        self.possible_agents = self.parallel_env.possible_agents

        # STATE STATISTICS
        self.state_stats = {}
        for agent in self.agents:
            state_stat = RunningStateStat(observations[agent])
            self.state_stats[agent] = state_stat

    def observation_space(self, agent):
        return self.parallel_env.observation_space(agent)

    def observation_size(self, agent):
        return self.parallel_env.observation_space(agent).shape[0]

    def action_space(self, agent):
        return self.parallel_env.action_space(agent)

    def action_size(self, agent):
        return self.parallel_env.action_space(agent).shape[0]

    def reset(self):
        observations = self.parallel_env.reset()
        observations_t = {agent: torch.tensor(observations[agent]) for agent in self.agents}
        observations_t = self.get_normalized(observations_t)
        return observations_t

    def step(self, actions_t):
        # ACTION TENSOR TO NUMPY
        actions = {agent: np.clip(actions_t[agent].detach().squeeze().numpy(), 0, 1) for agent in self.agents}
        # STEP
        try:
            observations, rewards, dones, infos = self.parallel_env.step(actions)
        except:
            pass
        # NUMPY TO TENSORS
        observations_t = {agent: torch.tensor(observations[agent]) for agent in self.agents}
        rewards_t = {agent: torch.tensor(rewards[agent]) for agent in self.agents}
        dones_t = {agent: torch.tensor(dones[agent]) for agent in self.agents}
        observations_t = self.get_normalized(observations_t)
        return observations_t, rewards_t, dones_t, infos

    def render(self, mode='human'):
        self.parallel_env.render(mode)

    def close(self):
        self.parallel_env.close()

    def seed(self, seed=None):
        self.parallel_env.seed(seed)

    def _prepare_action(self):
        pass

    def get_normalized(self, obs_tensor):
        for agent in self.agents:
            obs_np = obs_tensor[agent].detach().squeeze().numpy()
            self.update(agent, obs_np)
            obs_np = np.clip((obs_np - self.state_stats[agent].mean()) / (self.state_stats[agent].std() + 1e-6), -10.,
                             10.)
            output_state_tensor = torch.FloatTensor(obs_np)
            obs_tensor[agent] = output_state_tensor
        return obs_tensor

    def update(self, agent, state):
        self.state_stats[agent].len += 1
        old_mean = self.state_stats[agent].running_mean.copy()
        self.state_stats[agent].running_mean[...] = old_mean + (state - old_mean) / self.state_stats[agent].len
        self.state_stats[agent].running_std[...] = self.state_stats[agent].running_std + (state - old_mean) * (
                    state - self.state_stats[agent].running_mean)


class RunningStateStat:
    def __init__(self, state_np):
        # state_np = state_tensor.detach().squeeze().numpy()
        self.len = 1
        self.running_mean = state_np
        self.running_std = state_np ** 2

    def update(self, state):
        self.len += 1
        old_mean = self.running_mean.copy()
        self.running_mean[...] = old_mean + (state - old_mean) / self.len
        self.running_std[...] = self.running_std + (state - old_mean) * (state - self.running_mean)

    def mean(self):
        return self.running_mean

    def std(self):
        return np.sqrt(self.running_std / (self.len - 1))

    def get_normalized(self, state_tensor):
        state_np = state_tensor.detach().squeeze().numpy()
        self.update(state_np)
        state_np = np.clip((state_np - self.mean()) / (self.std() + 1e-6), -10., 10.)
        output_state_tensor = torch.FloatTensor(state_np)
        return output_state_tensor


class SingleAgentEnv:
    def __init__(self, env_name, plotter=None):
        self.env_name = env_name
        self.plotter = plotter
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.state_stat = RunningStateStat(self.env.reset())

    def reset(self):
        observation = self.env.reset()
        observation = torch.tensor(observation, requires_grad=True).float().unsqueeze(0)
        observation = self.state_stat.get_normalized(observation)
        return observation

    def render(self):
        self.env.render()

    def sample_action(self):
        action = self.env.action_space.sample()
        action = Variable(torch.tensor(action, requires_grad=True).float().unsqueeze(0))
        return action

    def sample_observation(self):
        observation = self.env.observation_space.sample()
        observation = torch.tensor(observation, requires_grad=True).float().unsqueeze(0)
        observation = self.state_stat.get_normalized(observation)
        return observation

    def step(self, action):
        action = self.prepare_action(action)
        observation, reward, done, info = self.env.step(action)
        observation = torch.tensor(observation, requires_grad=True).float().unsqueeze(0)
        reward = Variable(torch.tensor(reward).float().unsqueeze(0))
        done = torch.tensor(done)
        observation = self.state_stat.get_normalized(observation)
        return observation, reward, done, info

    def prepare_action(self, action):
        action = action.detach().squeeze().numpy()
        if self.env_name == "CartPole-v1":
            action = 1 if action > 0.5 else 0
            # print(action)
        elif self.env_name == "MountainCarContinuous-v0":
            action = [action]
        elif self.env_name == "LunarLanderContinuous-v2":
            action = action
        elif self.env_name == "BipedalWalker-v3":
            action = action
        else:
            if self.plotter:
                self.plotter.error('action!')
        return action

    def close(self):
        self.env.close()

    def observation_size(self):
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            return self.env.observation_space.n
        if isinstance(self.env.observation_space, gym.spaces.Box):
            return self.env.observation_space.shape[0]
        return None

    def action_size(self):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # self.parallel_env.action_space.n
            return 1
        if isinstance(self.env.action_space, gym.spaces.Box):
            return self.env.action_space.shape[0]
        return None
