import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from alg_GLOBALS import *
from alg_nets import *
from alg_plotter import ALGPlotter
from alg_env_wrapper import MultiAgentParallelEnvWrapper
from alg_env_demo import play_parallel_env, load_and_play


def train():
    best_score = - math.inf

    for i_update in range(N_UPDATES):

        with torch.no_grad():
            # SAMPLE TRAJECTORIES
            all_obs, all_actions, all_rewards, all_dones, all_next_obs, average_result_dict = get_trajectories()

            # COMPUTE RETURNS AND ADVANTAGES
            returns_t, advantages_t = compute_returns_and_advantages(all_obs, all_rewards, all_dones)

        # UPDATE CRITIC
        update_critic()

        # UPDATE ACTOR
        update_actors()

        # PLOTTER, NEPTUNE
        plotter.plot(i_update, average_result_dict)

        # RENDER, PRINT
        if i_update > N_UPDATES - 5:
            print(f'Update {i_update + 1}:')
            play_parallel_env(env, True, 1, actor_old)
        else:
            print(f'Update {i_update + 1}, average results: {average_result_dict} \n---')

        # SAVE
        average_score = sum(average_result_dict.values())
        if average_score > best_score:
            best_score = average_score
            save_results(SAVE_PATH, actor)

    env.close()
    plt.pause(0)


def get_trajectories():
    all_obs = {agent: [] for agent in env.agents}
    all_actions = {agent: [] for agent in env.agents}
    all_rewards = {agent: [] for agent in env.agents}
    all_dones = {agent: [] for agent in env.agents}
    all_next_obs = {agent: [] for agent in env.agents}
    average_result_dict = {agent: [] for agent in env.agents}

    n_episodes = 0
    step_count = 0

    while not step_count > BATCH_SIZE:
        observations_t = env.reset()
        episode_result_dict = {agent: 0 for agent in env.agents}
        while True:

            # action = get_train_action(actor_old, state)
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            actions_t = {agent: torch.tensor(actions[agent]) for agent in env.agents}
            next_observations_t, rewards_t, dones_t, infos = env.step(actions_t)

            for agent in env.agents:
                episode_result_dict[agent] += rewards_t[agent]
                all_obs[agent].append(observations_t[agent])
                all_actions[agent].append(actions_t[agent])
                all_rewards[agent].append(rewards_t[agent])
                all_dones[agent].append(dones_t[agent])
                all_next_obs[agent].append(next_observations_t[agent])

            observations_t = next_observations_t
            step_count += 1

            if False not in dones_t.values():
                break

        for agent in env.agents:
            average_result_dict[agent].append(episode_result_dict[agent])
        n_episodes += 1

    for agent in env.agents:
        average_result_dict[agent] = np.mean(average_result_dict[agent])
        all_obs[agent] = torch.stack(all_obs[agent])
        all_actions[agent] = torch.stack(all_actions[agent])
        all_rewards[agent] = torch.tensor(all_rewards[agent]) / n_episodes
        all_dones[agent] = torch.tensor(all_dones[agent])
        all_next_obs[agent] = torch.stack(all_next_obs[agent])

    return all_obs, all_actions, all_rewards, all_dones, all_next_obs, average_result_dict


def compute_returns_and_advantages(all_obs, all_rewards, all_dones):
    returns_t = {agent: torch.zeros(all_rewards[agent]) for agent in env.agents}
    deltas_t = {agent: torch.zeros(all_rewards[agent]) for agent in env.agents}
    advantages_t = {agent: torch.zeros(all_rewards[agent]) for agent in env.agents}
    critic_values_t = {agent: critic(all_obs[agent]).detach().squeeze() for agent in env.agents}

    for agent in env.agents:
        prev_return, prev_value, prev_advantage = 0, 0, 0
        for i in reversed(range(all_rewards[agent].shape[0])):
            final_state_bool = 1 - all_dones[agent][i]

            returns_t[agent][i] = all_rewards[agent][i] + GAMMA * prev_return * final_state_bool
            prev_return = returns_t[agent][i]

            deltas_t[agent][i] = all_rewards[agent][i] + GAMMA * prev_value * final_state_bool - critic_values_t[agent][i]
            prev_value = critic_values_t[agent][i]

            advantages_t[agent][i] = deltas_t[agent][i] + GAMMA * LAMBDA * prev_advantage * final_state_bool
            prev_advantage = advantages_t[agent][i]

        advantages_t[agent] = (advantages_t[agent] - advantages_t[agent].mean()) / (advantages_t[agent].std() + 1e-4)

    return returns_t, advantages_t


def update_critic():
    pass


def update_actors():
    pass


def save_results(path, model_to_save):
    # SAVE
    if SAVE_RESULTS:
        # SAVING...
        print(colored(f"Saving model...", 'green'))
        torch.save({
            'model': model_to_save,
            # 'len': env.state_stat.len,
            # 'mean': env.state_stat.running_mean,
            # 'std': env.state_stat.running_std,
        }, path)
    return path


if __name__ == '__main__':

    # --------------------------- # PARAMETERS # -------------------------- #
    N_UPDATES = 20
    BATCH_SIZE = 500  # size of the batches
    LR_CRITIC = 1e-3  # learning rate
    LR_ACTOR = 1e-3  # learning rate
    GAMMA = 0.995  # discount factor
    LAMBDA = 0.97

    # --------------------------- # CREATE ENV # -------------------------- #
    NUMBER_OF_AGENTS = 1
    # ENV = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=MAX_CYCLES, continuous_actions=True)
    ENV_NAME = 'simple_v2'
    ENV = simple_v2.parallel_env(max_cycles=25, continuous_actions=True)
    env = MultiAgentParallelEnvWrapper(ENV)

    NUMBER_OF_GAMES = 10

    # --------------------------- # NETS # -------------------------- #
    critic = CriticNet(obs_size=env.observation_size(env.agents[0]), n_actions=env.action_size(env.agents[0]), n_agents=1)
    actor = ActorNet(obs_size=env.observation_size(env.agents[0]), n_actions=env.action_size(env.agents[0]))
    actor_old = ActorNet(obs_size=env.observation_size(env.agents[0]), n_actions=env.action_size(env.agents[0]))
    # --------------------------- # OPTIMIZERS # -------------------------- #
    # critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)

    # --------------------------- # REPLAY BUFFER # -------------------------- #
    # replay_buffer = ReplayBuffer()

    # --------------------------- # FOR PLOT # -------------------------- #
    PLOT_PER = 1
    NEPTUNE = False
    PLOT_LIVE = True
    SAVE_RESULTS = True
    SAVE_PATH = f'data/actor_{ENV_NAME}.pt'
    plotter = ALGPlotter(plot_life=PLOT_LIVE, plot_neptune=NEPTUNE, name='my_run_ppo', tags=[ENV_NAME],
                         plot_per=PLOT_PER)

    # --------------------------- # PLOTTER INIT # -------------------------- #

    # --------------------------- # SEED # -------------------------- #
    SEED = 111
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    env.seed(SEED)
    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    # MAIN PROCESS
    train()

    # Example Plays
    print(colored('Example run...', 'green'))
    load_and_play(env, 1, SAVE_PATH)


# print(colored(f'\n~[WARNING]: {message}', 'yellow'), end=end)

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

























