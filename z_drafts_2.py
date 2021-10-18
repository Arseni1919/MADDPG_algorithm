from alg_env_wrapper import *


env = ALGEnv_Wrapper(ENV)

for i in range(10):
    print(i)
    for step in env.run_episode(render=True):
        pass