from alg_env_module import *


env = ALGEnv_Module(ENV)

for i in range(10):
    print(i)
    print(env.run_episode(render=True))
