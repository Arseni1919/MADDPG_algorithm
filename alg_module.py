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

        # FIRST INIT
        total_reward = 0
        steps_counter = 0

        for episode in range(M_EPISODES):

            # CHOOSE AN ACTION AND MAKE A STEP
            for step in env_module.run_episode(models_dict=self.actor_net_dict, render=False):
                experience, observations, actions, rewards, dones, new_observations = step
                total_reward += sum(rewards.values())

                # ADD TO EXPERIENCE BUFFER
                dm.train_dataset.append(experience)

                # TRAINING STEP
                self.training_step(steps_counter, step, dm)
                steps_counter += 1

            # END OF AN EPISODE
            plotter.debug(f"Finished episode {episode} with a total reward: {colored(f'{total_reward}', 'magenta')}.",
                          print_info=False)
            total_reward = 0
            self.validation_step(episode)
        plotter.info("Finished to fit the env.")

    def training_step(self, steps_counter, step, dm):
        if steps_counter % UPDATE_EVERY == 0:
            plotter.debug(f"TRAINING STEP (Step {steps_counter})")
            experience, observations, actions, rewards, dones, new_observations = step
            list_of_batches = list(dm.train_dataloader())
            n_batches_to_iterate = min(len(list_of_batches), BATCHES_PER_TRAINING_STEP)

            # FOR LOOP - BIG BATCHES
            for b_indx in range(n_batches_to_iterate):
                batch = list_of_batches[b_indx]
                observations, actions, rewards, dones, new_observations = batch

                # FOR LOOP - AGENTS
                for agent in env_module.get_agent_list():
                    observation = observations[agent]
                    action = actions[agent]
                    reward = rewards[agent]
                    done = dones[agent]
                    new_observation = new_observations[agent]

                    # COMPUTES TARGETS
                    y = self.get_y_targets(agent, reward, done, rewards, new_observations)

                    # UPDATES CRITIC
                    self.update_agent_critic(agent, y, observations, actions)

                    # UPDATES ACTOR
                    self.update_agent_actor(agent, done, observations, actions )

                # UPDATES TARGET NETWORK PARAMETERS
                self.update_target_net_params()

    def get_y_targets(self, curr_agent, agent_reward, agent_done, rewards, new_observations):

        next_target_actions, next_target_obs = [], []
        for agent in env_module.get_agent_list():
            new_observation = new_observations[agent]
            next_target_action = env_module.get_action(
                new_observation, agent_done, self.actor_target_net_dict[agent], noisy_action=True
            )
            next_target_actions.extend(next_target_action)
            next_target_obs.extend(new_observation)

        next_Q = self.critic_target_net_dict[curr_agent](next_target_obs.extend(next_target_actions))
        # y = agent_reward + GAMMA * (~agent_done) * next_Q
        y = sum(rewards) + GAMMA * (~agent_done) * next_Q
        # return torch.tensor([len(observations)])
        return y

    def update_agent_critic(self, curr_agent, y, observations, actions):
        actions_list, obs_list = [], []
        for agent in env_module.get_agent_list():
            obs_list.extend(observations[agent])
            actions_list.extend(actions[agent])

        q_value = self.critic_net_dict[curr_agent](obs_list.extend(actions_list))
        # critic_loss = nn.MSELoss()(Q_vals, y.detach())
        critic_loss = (y - q_value).pow(2).mean()
        self.critic_opts_dict[curr_agent].zero_grad()
        critic_loss.backward()
        self.critic_opts_dict[curr_agent].step()
        # plotter.warning("No update_agent_critic().")
        return critic_loss.item()

    def update_agent_actor(self, curr_agent, agent_done, observations, actions):
        actions_list, obs_list = [], []
        for agent in env_module.get_agent_list():
            obs_list.extend(observations[agent])
            if agent == curr_agent:
                curr_action = env_module.get_action(
                    observations[agent], agent_done, self.actor_net_dict[agent], noisy_action=True
                )
                actions_list.extend(curr_action)
            else:
                actions_list.extend(actions[agent])

        actor_loss = - self.critic_net_dict[curr_agent](obs_list.extend(actions_list)).mean()
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
        if episode % VAL_CHECKPOINT_INTERVAL == 0 and episode > 0:
            plotter.debug(f"VALIDATION STEP (episode {episode})")
            play(1, self.actor_net_dict, print_info=False, noisy_action=False)

    def configure_optimizers(self):
        pass

    def fit_sequential_env(self):
        pass

# # FIRST INIT
# observations = self.env.reset()
# dones = {agent: False for agent in self.env.agents}
# total_reward, episode = 0, 0
#
# for step in range(M_EPISODES * MAX_CYCLES):
#
#     # CHOOSE AN ACTION AND MAKE A STEP
#     actions = {agent: self.get_action_of_agent(agent, observations[agent], dones[agent]) for agent in self.env.agents}
#     new_observations, rewards, dones, infos = self.env.step(actions)
#     total_reward += sum(rewards.values())
#
# # ADD TO EXPERIENCE BUFFER experience = Experience(state=observations, action=actions, reward=rewards, done=dones,
# new_state=new_observations) dm.train_dataset.append(experience) observations = new_observations
#
#     if all(dones.values()):
#         # END OF AN EPISODE
#         episode += 1
#         print(f'{colored("finished", "green")} episode {episode} with a total reward: {total_reward}')
#
#         # FOLLOWING INIT AND VALIDATION STEP
#         total_reward = 0
#         observations = self.env.reset()
#         dones = {agent: False for agent in self.env.agents}
#         self.validation_step(episode)
#
#     # TRAINING STEP
#     self.training_step(step)
#
# self.env.close()
