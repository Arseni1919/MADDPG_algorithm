import time

from alg_constrants_amd_packages import *
from alg_general_functions import *


def alg_try(times=1, with_model=True):
    if with_model:
        model = torch.load('data/actor_net.pt')
        model.eval()
        models_dict = {agent: model for agent in env_module.get_agent_list()}
        play(times, models_dict=models_dict)
    else:
        play(times)


if __name__ == '__main__':
    print('Press enter..')
    for i in range(3):
        print("with model...")
        alg_try(with_model=True)
        time.sleep(1)
        print("no model...")
        alg_try(with_model=False)
        time.sleep(1)








