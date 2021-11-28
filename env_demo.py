import random

import pettingzoo

from alg_GLOBALS import *
from alg_env_wrapper import MultiAgentParallelEnvWrapper


def random_demo_parallel_env(parallel_env, render=True, episodes=10):
    max_cycles = 500
    for episode in range(episodes):
        observations = parallel_env.reset()
        result_dict = {agent: 0 for agent in parallel_env.agents}
        for step in range(max_cycles):
            actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
            observations, rewards, dones, infos = parallel_env.step(actions)
            for agent in parallel_env.agents:
                result_dict[agent] += rewards[agent]

            if render:
                parallel_env.render()
            if False not in dones.values():
                break

        print(f'Game {episode + 1} finished with result:')
        pprint(result_dict)
        print('---')
    parallel_env.close()


if __name__ == '__main__':
    # ENV = simple_v2.parallel_env(max_cycles=25, continuous_actions=True)
    ENV = simple_v2.parallel_env(max_cycles=25, continuous_actions=True)
    ENV = MultiAgentParallelEnvWrapper(ENV)

    # SEED
    SEED = 111
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    ENV.seed(SEED)

    episodes = 2
    # NOT FOR PARALLEL
    if isinstance(ENV, pettingzoo.AECEnv):
        print('Not parallel env')
        random_demo(ENV, render=True, episodes=episodes)
    if isinstance(ENV, pettingzoo.ParallelEnv) or isinstance(ENV, MultiAgentParallelEnvWrapper):
        print('Parallel env')
        random_demo_parallel_env(ENV, render=True, episodes=episodes)
