import random
import time

import pettingzoo

from alg_GLOBALS import *
from alg_env_wrapper import MultiAgentParallelEnvWrapper


def get_train_action(net, observation):
    action_mean, action_std = net(observation)
    action_dist = torch.distributions.Normal(action_mean, action_std)
    action = action_dist.sample()
    return action


def load_and_play(env_to_play, times, path_to_load_model):
    # Example runs
    curr_load_dict = torch.load(path_to_load_model)
    curr_models = curr_load_dict['models']
    env_to_play.state_stats = curr_load_dict['state_stats']
    # env_to_play.state_stat.running_mean = load_dict['mean']
    # env_to_play.state_stat.running_std = load_dict['std']
    # env_to_play.state_stat.len = load_dict['len']

    play_parallel_env(env_to_play, True, times, models=curr_models)


def play_parallel_env(parallel_env, render=True, episodes=10, models=None):
    max_cycles = 500
    for episode in range(episodes):
        observations = parallel_env.reset()
        result_dict = {agent: 0 for agent in parallel_env.agents}
        for step in range(max_cycles):
            if models:
                actions_t = {agent: get_train_action(models[agent], observations[agent]) for agent in parallel_env.agents}
                # actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
                # actions_t = {agent: torch.tensor(actions[agent]) for agent in parallel_env.agents}
                pass
            else:
                actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
                actions_t = {agent: torch.tensor(actions[agent]) for agent in parallel_env.agents}
            observations, rewards, dones, infos = parallel_env.step(actions_t)
            for agent in parallel_env.agents:
                result_dict[agent] += rewards[agent]

            if render:
                parallel_env.render()
                time.sleep(0.1)
            if False not in dones.values():
                break

        print(f'[{episode + 1}] Game finished with result:')
        pprint(result_dict)
        print('---')
    parallel_env.close()


if __name__ == '__main__':

    # load_dict = torch.load('data/actor_simple_v2.pt')
    # ENV = simple_v2.parallel_env(max_cycles=25, continuous_actions=True)

    ENV = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=50, continuous_actions=True)
    ENV = MultiAgentParallelEnvWrapper(ENV)
    load_dict = torch.load('data/actor_simple_spread_v2_3.pt')
    models = load_dict['models']
    ENV.state_stats = load_dict['state_stats']

    # SEED
    # SEED = 111
    # torch.manual_seed(SEED)
    # np.random.seed(SEED)
    # random.seed(SEED)
    # ENV.seed(SEED)

    n = 20
    # NOT FOR PARALLEL
    if isinstance(ENV, pettingzoo.AECEnv):
        print('Not parallel env')
        random_demo(ENV, render=True, episodes=n)
    if isinstance(ENV, pettingzoo.ParallelEnv) or isinstance(ENV, MultiAgentParallelEnvWrapper):
        print('Parallel env')
        play_parallel_env(ENV, render=True, episodes=n, models=models)


# PLAY EPISODE
# observations = env.reset()
# result_dict = {agent: 0 for agent in env.agents}
# while True:
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#     observations, rewards, dones, infos = env.step(actions)
#     for agent in env.agents:
#         result_dict[agent] += rewards[agent]
#
#     if False not in dones.values():
#         break
