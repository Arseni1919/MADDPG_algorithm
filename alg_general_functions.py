import pettingzoo

from alg_constrants_amd_packages import *
from alg_env_module import env_module


def get_action(env, agent, observation, done, model: nn.Module, step=0):
    with torch.no_grad():
        model_output = model.get_action(np.expand_dims(observation, axis=0))
        model_output = torch.squeeze(model_output)
        action = model_output.detach().numpy()
        action = env.action_space(agent).sample() if not done else None
        # noise = ACT_NOISE / np.log(step) if step > 5000 else ACT_NOISE
        # action = action + np.random.normal(0, noise, 2)
        # action = np.clip(action, -1, 1)
        return action


def get_actions(env, observations, dones, models):
    return {agent: get_action(env, agent, observations[agent], dones[agent], models[agent]) for agent in env.agents}


def play(times: int = 1, models=None):
    env = ENV

    if isinstance(env, pettingzoo.ParallelEnv):
        print('[Parallel Env]')
        total_reward, game = 0, 0
        for episode in range(times):
            for step in env_module.run_episode(models_dict=models, render=True):
                print('', end='')
                experience, observations, actions, rewards, dones, new_observations = step
                total_reward += sum(rewards.values())

            game += 1
            print(f'{colored("finished", "green")} game {game} with a total reward: {total_reward}')
            total_reward = 0

    if isinstance(env, pettingzoo.AECEnv):
        print('[AECEnv Env]')
        total_reward, game = 0, 0
        for episode in range(times):
            for step in env_module.run_episode_seq(models_dict=models, render=True):
                agent, observation, action, reward, done = step
                total_reward += reward
            game += 1
            print(f'{colored("finished", "green")} game {game} with a total reward: {total_reward}')
            total_reward = 0



