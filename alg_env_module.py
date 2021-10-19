from alg_constrants_amd_packages import *
from alg_plotter import plotter


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



class ALGEnv_Module:
    def __init__(self, env):
        self.env = env
        plotter.info("ALGEnv_Module instance created.")

    def get_agent_list(self):
        return self.env.unwrapped.agents

    def observation_space_shape(self, agent):
        # obs_size, n_actions = env_module.env.observation_space(agent), env_module.env.action_space(agent)
        obs_size = self.env.observation_space(agent)

        return torch.tensor(obs_size.shape)

    def action_space_shape(self, agent):
        n_actions = self.env.action_space(agent)
        return torch.tensor(n_actions.shape)

    def run_episode(self, models_dict=None, render=False, no_grad=True):

        with torch.no_grad():
            # INIT
            episode_ended = False
            observations = self.env.reset()
            dones = {agent: False for agent in self.env.agents}

            while not episode_ended:

                # CHOOSES ACTIONS
                if models_dict:
                    actions = get_actions(self.env, observations, dones, models_dict)
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


env_module = ALGEnv_Module(ENV)


