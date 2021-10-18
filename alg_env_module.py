from alg_constrants_amd_packages import *
from alg_general_functions import *


class ALGEnv_Module:
    def __init__(self, env):
        self.env = env

    def run_episode(self, models_dict=None, render=False, no_grad=True):

        with torch.no_grad():
            episode_ended = False
            observations = self.env.reset()
            dones = {agent: False for agent in self.env.agents}

            while not episode_ended:
                if models_dict:
                    actions = get_actions(self.env, observations, dones, models_dict)
                else:
                    actions = {agent: ENV.action_spaces[agent].sample() for agent in self.env.agents}
                new_observations, rewards, dones, infos = ENV.step(actions)

                # ADD TO EXPERIENCE BUFFER
                experience = Experience(state=observations, action=actions, reward=rewards, done=dones,
                                        new_state=new_observations)
                yield experience, observations, actions, rewards, dones, new_observations
                observations = new_observations

                if render:
                    self.env.render()

                if all(dones.values()):
                    observations = self.env.reset()
                    dones = {agent: False for agent in self.env.agents}
                    # print(f'{colored("finished", "green")} game {game} with a total reward: {total_reward}')
                    episode_ended = True

            self.env.close()

    def run_episode_seq(self, models_dict=None, render=False):

        with torch.no_grad():
            self.env.reset()
            for agent in self.env.agent_iter():
                observation, reward, done, info = self.env.last()
                if models_dict:
                    action = get_action(self.env, agent, observation, done, models_dict[agent])
                else:
                    action = self.env.action_spaces[agent].sample()
                action = action if not done else None
                self.env.step(action)
                yield agent, observation, action, reward, done

                if render:
                    self.env.render()

            self.env.close()


env_module = ALGEnv_Module(ENV)


