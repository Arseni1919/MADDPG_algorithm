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
    ):
        self.env = env

        self.critic_net_dict = critic_net_dict
        self.critic_target_net_dict = critic_target_net_dict
        self.actor_net_dict = actor_net_dict

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
                    y = self.get_y_targets(reward, done, observations, actions, new_observations)

                    # UPDATES CRITIC
                    self.update_agent_critic()

                    # UPDATES ACTOR
                    self.update_agent_actor()

                # UPDATES TARGET NETWORK PARAMETERS
                self.update_target_net_params()

    def get_y_targets(self, agent_reward, agent_done, observations, actions, new_observations):
        means, stds = self.actor_net(next_states)

        normal_dist = Normal(loc=torch.zeros(means.shape), scale=torch.ones(means.shape))
        new_actions = torch.tanh(means + torch.mul(stds, normal_dist.sample()))

        Q_target_vals_1 = self.critic_target_net_1(next_states, new_actions)
        Q_target_vals_2 = self.critic_target_net_2(next_states, new_actions)
        min_Q_vals = torch.minimum(Q_target_vals_1, Q_target_vals_2)

        normal_dist = Normal(loc=means, scale=stds)
        log_probs = normal_dist.log_prob(new_actions)
        log_probs = torch.nan_to_num(log_probs)
        log_policy_a_s = log_probs - torch.log(1 - new_actions.pow(2))
        log_policy_a_s = torch.sum(log_policy_a_s, dim=1)
        subtraction = torch.sub(torch.squeeze(min_Q_vals), log_policy_a_s, alpha=ALPHA)
        return rewards.float() + GAMMA * (~dones).float() * subtraction
        return torch.tensor([len(observations)])

    def update_agent_critic(self):
        # update critic - gradient descent
        # critic_losses = []
        # for i in range(len(self.critic_nets)):
        #     self.critic_opts[i].zero_grad()
        #     Q_vals = self.critic_nets[i](states, actions)
        #     Q_vals = torch.squeeze(Q_vals)
        #     critic_loss = nn.MSELoss()(Q_vals, y.detach())
        #     critic_loss.backward()
        #     self.critic_opts[i].step()
        #     critic_losses.append(critic_loss)
        # plotter.warning("No update_agent_critic().")
        pass

    def update_agent_actor(self):
        # update actor - gradient ascent
        # actor_loss = self.execute_policy_gradient_ascent(states, actions, rewards, dones, next_states)
        # self.actor_opt.zero_grad()
        # means, stds = self.actor_net(states)
        # normal_dist = Normal(loc=means, scale=stds)
        # new_actions_before_tanh = normal_dist.rsample()
        # new_actions = torch.tanh(new_actions_before_tanh)
        #
        # # minimum out of Q functions
        # Q_target_vals_1 = self.critic_target_net_1(next_states, new_actions)
        # Q_target_vals_2 = self.critic_target_net_2(next_states, new_actions)
        # min_Q_vals = torch.minimum(Q_target_vals_1, Q_target_vals_2)
        #
        # # log of policy of a certain action given certain state
        # first_term = normal_dist.log_prob(new_actions_before_tanh)
        # second_term = torch.log(1 - new_actions.pow(2) + EPSILON)
        # log_policy_a_s_before_sum = first_term - second_term
        # log_policy_a_s = torch.sum(log_policy_a_s_before_sum, dim=1)
        #
        # subtraction = torch.sub(torch.squeeze(min_Q_vals), log_policy_a_s, alpha=ALPHA)
        # actor_loss = - subtraction.mean()
        # actor_loss.backward()
        # self.actor_opt.step()
        # return actor_loss

        # PLOTS PROCESS
        # plotter.plot_online({'actor_loss': actor_loss.item(), 'b_indx': b_indx, })
        # plotter.warning("No update_agent_actor().")
        pass

    def update_target_net_params(self):
        # update target networks
        for agent in env_module.get_agent_list():
            for target_param, param in zip(self.critic_target_net_dict[agent].parameters(),
                                           self.critic_net_dict[agent].parameters()):
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
