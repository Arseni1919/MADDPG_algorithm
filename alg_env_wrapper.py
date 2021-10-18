from alg_constrants_amd_packages import *
from alg_general_functions import *


class ALGEnv_Wrapper:
    def __init__(self, env):
        self.env = env

    def run_episode(self, models_dict=None, render=False):
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
                    episode_ended = True


            self.env.close()

