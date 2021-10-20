import pettingzoo

from alg_constrants_amd_packages import *
from alg_env_module import env_module
from alg_plotter import plotter


def play(times: int = 1, models_dict=None, print_info=True, noisy_action=False):
    env = ENV
    plotter.info(f"Playing the game {times} times...", print_info)

    if isinstance(env, pettingzoo.ParallelEnv):
        plotter.debug("Inside the Parallel Env.", print_info)
        total_reward, game = 0, 0
        for episode in range(times):
            for step in env_module.run_episode(models_dict=models_dict, render=True, noisy_action=noisy_action):
                print('', end='')
                experience, observations, actions, rewards, dones, new_observations = step
                total_reward += sum(rewards.values())

            game += 1
            plotter.info(f"Finished game {game} with a total reward: {colored(f'{total_reward}', 'magenta')}.", print_info)
            total_reward = 0

    if isinstance(env, pettingzoo.AECEnv):
        plotter.debug("Inside the AECEnv Env.", print_info)
        total_reward, game = 0, 0
        for episode in range(times):
            for step in env_module.run_episode_seq(models_dict=models_dict, render=True):
                agent, observation, action, reward, done = step
                total_reward += reward
            game += 1
            plotter.info(f"Finished game {game} with a total reward: {colored(f'{total_reward}', 'magenta')}.", print_info)
            total_reward = 0



