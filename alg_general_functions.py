import pettingzoo

from alg_constrants_amd_packages import *


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
        with torch.no_grad():
            observations = env.reset()
            dones = {agent: False for agent in env.agents}
            total_reward, game = 0, 0
            for step in range(times * MAX_CYCLES):
                if models:
                    actions = get_actions(env, observations, dones, models)
                else:
                    actions = {agent: ENV.action_spaces[agent].sample() for agent in env.agents}
                observations, rewards, dones, infos = ENV.step(actions)
                # print(f'{env.agents}: {dones}, {rewards}, {observations}')
                total_reward += sum(rewards.values())
                if all(dones.values()):
                    observations = env.reset()
                    dones = {agent: False for agent in env.agents}
                    game += 1
                    print(f'{colored("finished", "green")} game {game} with a total reward: {total_reward}')
                    total_reward = 0
                env.render()
            env.close()

    if isinstance(env, pettingzoo.AECEnv):
        print('[AECEnv Env]')
        with torch.no_grad():
            for game in range(times):
                env.reset()
                total_reward = 0
                for agent in env.agent_iter():
                    observation, reward, done, info = env.last()
                    if models:
                        action = get_action(env, agent, observation, done, models[agent])
                    else:
                        action = env.action_spaces[agent].sample()
                    action = action if not done else None
                    env.step(action)
                    total_reward += reward
                    # print(f'{agent}: {done}, {reward}, {observation}')
                    env.render()
                print(f'{colored("finished", "green")} game {game} with a total reward: {total_reward}')
                total_reward = 0
            env.close()


