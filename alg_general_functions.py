import pettingzoo

from alg_constrants_amd_packages import *



def get_action(state, model: nn.Module, step=0):
    with torch.no_grad():
        model_output = model.get_action(np.expand_dims(state, axis=0))
        model_output = torch.squeeze(model_output)
        action = model_output.detach().numpy()
        # noise = ACT_NOISE / np.log(step) if step > 5000 else ACT_NOISE
        # action = action + np.random.normal(0, noise, 2)
        # action = np.clip(action, -1, 1)
        return action


def fill_the_buffer(train_dataset, env, actor_net):
    state = env.reset()
    while len(train_dataset) < UPDATE_AFTER:
        action = get_action(state, actor_net)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        experience = Experience(state=state, action=action, reward=reward, done=done, new_state=next_state)
        train_dataset.append(experience)
        if done:
            state = env.reset()

    env.close()


def play(times: int = 1, models=None):
    env = ENV

    if isinstance(env, pettingzoo.ParallelEnv):
        print('[Parallel Env]')
        with torch.no_grad():
            observations = env.reset()
            total_reward, game = 0, 0
            for step in range(times * MAX_CYCLES):
                if models:
                    actions = get_action(observations, models)
                else:
                    actions = {agent: ENV.action_spaces[agent].sample() for agent in env.agents}
                observations, rewards, dones, infos = ENV.step(actions)
                # print(f'{env.agents}: {dones}, {rewards}, {observations}')
                total_reward += sum(rewards.values())
                if all(dones.values()):
                    observations = env.reset()
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
                        action = get_action(observation, models)
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


