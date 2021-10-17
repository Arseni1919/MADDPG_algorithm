from alg_constrants_amd_packages import *

from pettingzoo.mpe import simple_spread_v2
env = simple_spread_v2.env(N=3, local_ratio=0.5, max_cycles=25)

# from pettingzoo.utils import random_demo
# random_demo(env, render=True, episodes=1)

for i in range(10):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        # action = policy(observation, agent)
        action = env.action_spaces[agent].sample() if not done else None
        env.step(action)
        env.render()