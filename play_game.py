from alg_constrants_amd_packages import *
from alg_general_functions import *


def alg_try(with_model=True):
    if with_model:
        model = torch.load('data/actor_net.pt')
        model.eval()
        models_dict = {agent: model for agent in env_module.get_agent_list()}
        play(10, models_dict=models_dict)
    else:
        play(10)


if __name__ == '__main__':
    alg_try(with_model=False)








