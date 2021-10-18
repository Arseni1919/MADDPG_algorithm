from alg_constrants_amd_packages import *
from alg_general_functions import *
from alg_memory import ALGDataset
from alg_nets import ActorNet, CriticNet
from alg_plotter import plotter


class ALGModule:
    def __init__(
            self,
            env: pettingzoo.AECEnv,
            critic_net_list: list[CriticNet],
            critic_target_net_list: list[CriticNet],
            actor_net_list: list[ActorNet],
            train_dataset: ALGDataset,
            train_dataloader: torch.utils.data.DataLoader
    ):
        self.env = env

        self.critic_net_list = critic_net_list
        self.critic_target_net_list = critic_target_net_list

        self.actor_net_list = actor_net_list

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader

        # create optimizers
        self.critic_opts = [torch.optim.Adam(net.parameters(), lr=LR_CRITIC) for net in self.critic_net_list]
        self.actor_opts = [torch.optim.Adam(net.parameters(), lr=LR_ACTOR) for net in self.actor_net_list]

    def fit(self):
        # random process -> torch.normal(mean=torch.tensor(10.0), std=torch.tensor(10.0))

        for episode in range(M_EPISODES):
            self.validation_step(episode)
            observation = self.env.reset()
            total_reward = 0
            for agent in self.env.agent_iter():
                observation, reward, done, info = self.env.last()
                action = self.get_action_of_agent(agent, observation, done)
                self.env.step(action)
                total_reward += reward
                # self.render()
                self.training_step(agent)

            print(f'finished episode {episode} with a total reward: {total_reward}')

        self.env.close()

        for step in range(MAX_STEPS):
            self.validation_step(step)

            action = get_action(state, self.actor_net, step)
            next_state, reward, done, _ = self.env.step(action)

            experience = Experience(state=state, action=action, reward=reward, done=done, new_state=next_state)
            self.train_dataset.append(experience)

            state = next_state if done else self.env.reset()

            self.training_step(step)

        self.env.close()

    def training_step(self, step):
        if step % UPDATE_EVERY == 0:
            print(f'[TRAINING STEP] Step: {step}')

            list_of_batches = list(self.train_dataloader)
            n_batches_to_iterate = min(len(list_of_batches), BATCHES_IN_TRAINING_STEP)

            for b_indx in range(n_batches_to_iterate):  # range(len(list_of_batches)) | range(x)
                batch = list_of_batches[b_indx]
                states, actions, rewards, dones, next_states = batch

                # compute targets
                y = self.get_y_targets(rewards, dones, next_states)

                # update critic - gradient descent
                critic_losses = []
                for i in range(len(self.critic_nets)):
                    self.critic_opts[i].zero_grad()
                    Q_vals = self.critic_nets[i](states, actions)
                    Q_vals = torch.squeeze(Q_vals)
                    critic_loss = nn.MSELoss()(Q_vals, y.detach())
                    critic_loss.backward()
                    self.critic_opts[i].step()
                    critic_losses.append(critic_loss)

                # update actor - gradient ascent
                actor_loss = self.execute_policy_gradient_ascent(states, actions, rewards, dones, next_states)

                # update target networks
                for i in range(len(self.critic_nets)):
                    for target_param, param in zip(self.critic_target_nets[i].parameters(), self.critic_nets[i].parameters()):
                        target_param.data.copy_(POLYAK * target_param.data + (1.0 - POLYAK) * param.data)

                plotter.plot_online(
                    {
                        'actor_loss': actor_loss.item(),
                        'critic_loss_1': critic_losses[0].item(),
                        'critic_loss_2': critic_losses[1].item(),
                        'b_indx': b_indx,
                    }
                )
                plotter.neptune_update(loss=None)

    def get_y_targets(self, rewards, dones, next_states):
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

    def execute_policy_gradient_ascent(self, states, actions, rewards, dones, next_states):
        self.actor_opt.zero_grad()
        means, stds = self.actor_net(states)
        normal_dist = Normal(loc=means, scale=stds)
        new_actions_before_tanh = normal_dist.rsample()
        new_actions = torch.tanh(new_actions_before_tanh)

        # minimum out of Q functions
        Q_target_vals_1 = self.critic_target_net_1(next_states, new_actions)
        Q_target_vals_2 = self.critic_target_net_2(next_states, new_actions)
        min_Q_vals = torch.minimum(Q_target_vals_1, Q_target_vals_2)

        # log of policy of a certain action given certain state
        first_term = normal_dist.log_prob(new_actions_before_tanh)
        second_term = torch.log(1 - new_actions.pow(2) + EPSILON)
        log_policy_a_s_before_sum = first_term - second_term
        log_policy_a_s = torch.sum(log_policy_a_s_before_sum, dim=1)

        subtraction = torch.sub(torch.squeeze(min_Q_vals), log_policy_a_s, alpha=ALPHA)
        actor_loss = - subtraction.mean()
        actor_loss.backward()
        self.actor_opt.step()
        return actor_loss

    def validation_step(self, episode):
        if episode % VAL_CHECKPOINT_INTERVAL == 0 and episode > 0:
            print(f'[VALIDATION STEP] Episode: {episode}')
            play(1, self.actor_net_list)

    def configure_optimizers(self):
        pass

    def get_action_of_agent(self, agent, observation, done):
        agent_model = self.actor_net_list[agent]
        action = agent_model(observation) if not done else None
        return action

    def render(self):
        if RENDER_WHILE_TRAINING:
            self.env.render()
