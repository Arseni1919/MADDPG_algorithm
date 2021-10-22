import torch

from alg_constrants_amd_packages import *
from alg_plotter import plotter


class ALGEnvModule:
    def __init__(self, env, noise_std=.3):
        self.env = env
        self.noise_std = noise_std
        plotter.info("ALGEnvModule instance created.")

    def get_action(self, observation, done, model: nn.Module, noisy_action=True):
        # AGENT OUT OF A GAME
        if done:
            return torch.zeros(size=list(self.env.action_spaces.values())[0].shape)

        # GETS THE ACTION
        observation = Variable(torch.from_numpy(np.expand_dims(observation, axis=0)).float().unsqueeze(0))
        model_output = model(observation)
        model_output = torch.squeeze(model_output)
        action = model_output.float().detach().numpy()

        # IF NO NEED TO ADD SOME RANDOM NOISE
        if noisy_action:
            # ADDS NOISE
            #           (1) random process -> torch.normal(mean=torch.tensor(10.0), std=torch.tensor(10.0))
            #           (2) m = Normal(torch.tensor([0.0]), torch.tensor([1.0])) -> m.sample()
            action += torch.normal(mean=torch.tensor(0.0), std=torch.tensor(self.noise_std)).item()

        clipped_action = np.clip(action, 0, 1)

        return clipped_action

    def get_action_no_grad(self, observation, done, model: nn.Module, noisy_action=True):
        with torch.no_grad():
            return self.get_action(observation, done, model, noisy_action)

    def get_actions(self, env, observations, dones, models, noisy_action=True):
        return {
            agent: self.get_action_no_grad(observations[agent], dones[agent], models[agent], noisy_action=noisy_action)
            for agent in env.agents
        }

    def get_agent_list(self):
        return self.env.unwrapped.agents

    def observation_space_shape(self, agent):
        # obs_size, n_actions = env_module.env.observation_space(agent), env_module.env.action_space(agent)
        obs_size = self.env.observation_space(agent)

        return torch.tensor(obs_size.shape)

    def action_space_shape(self, agent):
        n_actions = self.env.action_space(agent)
        return torch.tensor(n_actions.shape)

    def run_episode(self, models_dict=None, render=False, noisy_action=True):
        # INIT
        episode_ended = False
        observations = self.env.reset()
        dones = {agent: False for agent in self.env.agents}

        while not episode_ended:

            # CHOOSES ACTIONS
            if models_dict:
                actions = self.get_actions(self.env, observations, dones, models_dict, noisy_action=noisy_action)
            else:
                actions = {agent: ENV.action_spaces[agent].sample() for agent in self.env.agents}

            # TAKES THE ACTIONS INSIDE ENV
            new_observations, rewards, dones, infos = ENV.step(actions)

            # ADDS TO EXPERIENCE BUFFER
            experience = Experience(state=observations, action=actions, reward=rewards, done=dones,
                                    new_state=new_observations)
            # YIELDS INFO
            yield experience, observations, actions, rewards, dones, new_observations

            # UPDATES OBSERVATIONS VARIABLE
            observations = new_observations

            if render:
                self.env.render()

            if all(dones.values()):
                # END OF THE EPISODE
                observations = self.env.reset()
                dones = {agent: False for agent in self.env.agents}
                # print(f'{colored("finished", "green")} game {game} with a total reward: {total_reward}')
                episode_ended = True

        self.env.close()

    def run_episode_seq(self, models_dict=None, render=False):
        # with torch.no_grad():
        #     self.env.reset()
        #     for agent in self.env.agent_iter():
        #         observation, reward, done, info = self.env.last()
        #         if models_dict:
        #             action = get_action(self.env, agent, observation, done, models_dict[agent])
        #         else:
        #             action = self.env.action_spaces[agent].sample()
        #         action = action if not done else None
        #         self.env.step(action)
        #         yield agent, observation, action, reward, done
        #
        #         if render:
        #             self.env.render()
        #
        #     self.env.close()
        pass


env_module = ALGEnvModule(ENV, noise_std=ACT_NOISE)


