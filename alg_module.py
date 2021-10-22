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
                self.training_step(steps_counter, dm)
                steps_counter += 1

            # END OF AN EPISODE
            plotter.debug(f"Finished episode {episode} with a total reward: {colored(f'{total_reward}', 'magenta')}.",
                          print_info=False)
            total_reward = 0
            self.validation_step(episode)

        plotter.info("Finished to fit the env.")

    def training_step(self, steps_counter, dm, step=None):
        # experience, observations, actions, rewards, dones, new_observations = step
        if steps_counter % TRAIN_EVERY == 0:
            plotter.debug(f"TRAINING STEP (Step {steps_counter})")
            for curr_agent in env_module.get_agent_list():
                # RANDOM MINIBATCH
                observations, actions, rewards, dones, new_observations = dm.train_dataloader()[0]
                # COMPUTES TARGETS
                y = self.get_y_targets(curr_agent, rewards, dones, new_observations)
                # UPDATES CRITIC
                critic_loss = self.update_agent_critic(curr_agent, y, observations, actions)
                # UPDATES ACTOR
                actor_loss = self.update_agent_actor(curr_agent, observations, actions, dones)

            # UPDATES TARGET NETWORK PARAMETERS
            self.update_target_net_params()

            # list_of_batches = list(dm.train_dataloader())
            # n_batches_to_iterate = min(len(list_of_batches), BATCHES_PER_TRAINING_STEP)
            #
            # # FOR LOOP - BIG BATCHES
            # for b_indx in range(n_batches_to_iterate):
            #     batch = list_of_batches[b_indx]
            #     observations, actions, rewards, dones, new_observations = batch
            #
            #     # FOR LOOP - AGENTS
            #     for curr_agent in env_module.get_agent_list():
            #         observation = observations[curr_agent]
            #         action = actions[curr_agent]
            #         reward = rewards[curr_agent]
            #         done = dones[curr_agent]
            #         new_observation = new_observations[curr_agent]
            #
            #         # COMPUTES TARGETS
            #         y = self.get_y_targets(curr_agent, reward, done, rewards, new_observations)
            #
            #         # UPDATES CRITIC
            #         self.update_agent_critic(curr_agent, y, observations, actions)
            #
            #         # UPDATES ACTOR
            #         self.update_agent_actor(curr_agent, done, observations, actions )
            #
            #     # UPDATES TARGET NETWORK PARAMETERS
            #     self.update_target_net_params()

    def get_y_targets(self, curr_agent, rewards, dones, new_observations):
        y = []
        for j in range(len(dones)):
            next_target_actions, next_target_obs = [], []
            j_new_observations = new_observations[j]
            j_dones = dones[j]
            for agent in env_module.get_agent_list():
                next_target_obs.extend(j_new_observations[agent])
                next_target_action = env_module.get_action_no_grad(
                    j_new_observations[agent], j_dones[agent], self.actor_target_net_dict[agent], noisy_action=True
                )
                next_target_actions.extend(next_target_action)
            next_Q_j_curr_agent = self.critic_target_net_dict[curr_agent](next_target_obs.extend(next_target_actions))
            y_j = rewards[j][curr_agent] + GAMMA * (~dones[j][curr_agent]) * next_Q_j_curr_agent
            y.append(y_j)
        return y

    def update_agent_critic(self, curr_agent, y, observations, actions):
        loss = nn.MSELoss()
        input_list = []
        for j in range(len(y)):
            actions_list, obs_list = [], []
            for agent in env_module.get_agent_list():
                obs_list.extend(observations[j][agent])
                actions_list.extend(actions[j][agent])
            Q_j_curr_agent = self.critic_net_dict[curr_agent](obs_list.extend(actions_list))
            input_list.append(Q_j_curr_agent)
        critic_loss = loss(input_list, y)
        self.critic_opts_dict[curr_agent].zero_grad()
        critic_loss.backward()
        self.critic_opts_dict[curr_agent].step()
        return critic_loss.item()

    def update_agent_actor(self, curr_agent, observations, actions, dones):
        input_list = []
        for j in range(len(observations)):
            actions_list, obs_list = [], []
            for agent in env_module.get_agent_list():
                obs_list.extend(observations[j][agent])
                if agent != curr_agent:
                    actions_list.extend(actions[j][agent])
                else:
                    curr_action = env_module.get_action(
                        observation=observations[j][agent],
                        done=dones[j][agent],
                        model=self.actor_net_dict[agent],
                        noisy_action=True
                    )
                    actions_list.extend(curr_action)
            Q_j_curr_agent = self.critic_net_dict[curr_agent](obs_list.extend(actions_list))
            input_list.append(Q_j_curr_agent)
        actor_loss = - torch.tensor(input_list).mean()

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
        if episode % VAL_EVERY == 0 and episode > 0:
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
#     b_all_new_observations, b_all_rewards, dones, infos = self.env.step(actions)
#     total_reward += sum(b_all_rewards.values())
#
# # ADD TO EXPERIENCE BUFFER experience = Experience(state=observations, action=actions, reward=b_all_rewards, done=dones,
# new_state=b_all_new_observations) dm.train_dataset.append(experience) observations = b_all_new_observations
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
