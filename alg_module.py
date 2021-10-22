import torch

from alg_constrants_amd_packages import *
from alg_general_functions import *
from alg_nets import ActorNet, CriticNet
from alg_plotter import plotter


class ALGModule:
    def __init__(
            self,
            env,
            critic_net_dict,
            critic_target_net_dict,
            actor_net_dict,
            actor_target_net_dict,
    ):
        self.env = env

        self.critic_net_dict = critic_net_dict
        self.critic_target_net_dict = critic_target_net_dict
        self.actor_net_dict = actor_net_dict
        self.actor_target_net_dict = actor_target_net_dict

        # create optimizers
        self.critic_opts_dict = {agent: torch.optim.Adam(net.parameters(), lr=LR_CRITIC) for agent, net in
                                 self.critic_net_dict.items()}
        self.actor_opts_dict = {agent: torch.optim.Adam(net.parameters(), lr=LR_ACTOR) for agent, net in
                                self.actor_net_dict.items()}
        plotter.info("ALGModule instance created.")

    def fit(self, dm):
        if isinstance(self.env, pettingzoo.ParallelEnv):
            plotter.debug("Fitting the Parallel Env.")
            self.fit_parallel_env(dm)
        elif isinstance(self.env, pettingzoo.AECEnv):
            plotter.debug("Fitting the AECEnv Env.")
            self.fit_sequential_env()
        else:
            plotter.error("The instance of env is unknown.")

    def fit_parallel_env(self, dm):
        plotter.info("Beginning to fit the env...")
        plotter.plots_set(env_module=env_module)

        # FIRST INIT
        total_reward = 0
        steps_counter = 0

        for episode in range(M_EPISODES):
            plotter.info(f'\rEpisode {episode}..', end='')

            # CHOOSE AN ACTION AND MAKE A STEP
            for step in env_module.run_episode(models_dict=self.actor_net_dict, render=False):
                experience, observations, actions, rewards, dones, new_observations = step
                total_reward += sum(rewards.values())

                # ADD TO EXPERIENCE BUFFER
                dm.train_dataset.append(experience)

                # TRAINING STEP
                self.training_step(steps_counter, dm)
                steps_counter += 1

            # END OF AN EPISODE
            plotter.debug(f"Finished episode {episode} with a total reward: {colored(f'{total_reward}', 'magenta')}.",
                          print_info=False)
            plotter.plots_update_data({'reward': total_reward}, 'total_reward')
            plotter.plots_online()
            total_reward = 0
            self.validation_step(episode)

        plotter.info("\rFinished to fit the env.")

    def training_step(self, steps_counter, dm, step=None):
        # experience, observations, actions, rewards, dones, new_observations = step
        if steps_counter % TRAIN_EVERY == 0:
            # plotter.debug(f"\rTRAINING STEP (Step {steps_counter})", end='')
            for curr_agent in env_module.get_agent_list():
                # RANDOM MINIBATCH
                observations, actions, rewards, dones, new_observations = list(dm.train_dataloader())[0]
                # COMPUTES TARGETS
                y = self.get_y_targets(curr_agent, rewards, dones, new_observations)
                # UPDATES CRITIC
                critic_loss = self.update_agent_critic(curr_agent, y, observations, actions)
                # UPDATES ACTOR
                actor_loss = self.update_agent_actor(curr_agent, observations, actions, dones)

                plotter.plots_update_data({curr_agent: critic_loss}, 'critic')
                plotter.plots_update_data({curr_agent: actor_loss}, 'actor')
            # UPDATES TARGET NETWORK PARAMETERS
            self.update_target_net_params()

    def get_y_targets(self, curr_agent, rewards, dones, new_observations):
        y = []
        for j in range(BATCH_SIZE):
            next_target_actions, next_target_obs = [], []
            for agent in env_module.get_agent_list():
                next_target_obs.extend(new_observations[agent][j])
                next_target_action = env_module.get_action_no_grad(
                    new_observations[agent][j], dones[agent][j], self.actor_target_net_dict[agent], noisy_action=True
                )
                next_target_actions.extend(next_target_action)
            next_Q_j_curr_agent = self.critic_target_net_dict[curr_agent](next_target_obs, next_target_actions)
            y_j = rewards[curr_agent][j] + GAMMA * (~dones[curr_agent][j]) * next_Q_j_curr_agent
            y.append(y_j.squeeze())
        y = torch.tensor(y)
        return y

    def update_agent_critic(self, curr_agent, y, observations, actions):
        loss = nn.MSELoss()
        input_list = []
        for j in range(BATCH_SIZE):
            actions_list, obs_list = [], []
            for agent in env_module.get_agent_list():
                obs_list.extend(observations[agent][j])
                actions_list.extend(actions[agent][j])
            Q_j_curr_agent = self.critic_net_dict[curr_agent](obs_list, actions_list)
            input_list.append(Q_j_curr_agent.squeeze())

        input_list = torch.stack(input_list)
        critic_loss = loss(input_list, y)
        self.critic_opts_dict[curr_agent].zero_grad()
        critic_loss.backward()
        self.critic_opts_dict[curr_agent].step()
        return critic_loss.item()

    def update_agent_actor(self, curr_agent, observations, actions, dones):
        input_list = []
        for j in range(BATCH_SIZE):
            actions_list, obs_list = [], []
            for agent in env_module.get_agent_list():
                obs_list.extend(observations[agent][j])
                if agent != curr_agent:
                    actions_list.extend(actions[agent][j])
                else:
                    curr_action = env_module.get_action(
                        observation=observations[agent][j],
                        done=dones[agent][j],
                        model=self.actor_net_dict[agent],
                        noisy_action=True
                    )
                    actions_list.extend(curr_action)
            Q_j_curr_agent = self.critic_net_dict[curr_agent](obs_list, actions_list)
            input_list.append(Q_j_curr_agent)
        actor_loss = - torch.stack(input_list).mean()

        self.actor_opts_dict[curr_agent].zero_grad()
        actor_loss.backward()
        self.actor_opts_dict[curr_agent].step()
        return actor_loss.item()

    def update_target_net_params(self):
        # SOFT UPDATE
        for agent in env_module.get_agent_list():
            # CRITIC SOFT UPDATE
            for target_param, param in zip(self.critic_target_net_dict[agent].parameters(),
                                           self.critic_net_dict[agent].parameters()):
                target_param.data.copy_(POLYAK * target_param.data + (1.0 - POLYAK) * param.data)
            # ACTOR SOFT UPDATE
            for target_param, param in zip(self.actor_target_net_dict[agent].parameters(),
                                           self.actor_net_dict[agent].parameters()):
                target_param.data.copy_(POLYAK * target_param.data + (1.0 - POLYAK) * param.data)

    def validation_step(self, episode):
        if episode % VAL_EVERY_EPISODE == 0 and episode > 0:
            plotter.debug(f"\rVALIDATION STEP (episode {episode})", end='')
            total_rewards = play(1, self.actor_net_dict, print_info=True, noisy_action=False)
            plotter.plots_update_data({'reward': total_rewards[0]}, 'val_reward')

    def configure_optimizers(self):
        pass

    def fit_sequential_env(self):
        pass
