from alg_constrants_amd_packages import *
from alg_general_functions import *
from alg_nets import *
from alg_data_module import *
from alg_module import *


def train():
    # Initialization

    # ENV
    env =ENV
    obs_size = env.observation_spaces.keys()[0]
    n_actions = env.action_spaces.keys()[0]

    # NETS
    critic_net_dict, critic_target_net_dict, actor_net_dict = {}, {}, {}
    for agent in env.agents:

        obs_size, n_actions = env.observation_space(agent), env.action_space(agent)

        critic_net_i = CriticNet(obs_size, n_actions)
        critic_target_net_i = CriticNet(obs_size, n_actions)
        critic_target_net_i.load_state_dict(critic_net_i.state_dict())
        actor_net = ActorNet(obs_size, n_actions)

        critic_net_dict[agent] = critic_net_i
        critic_target_net_dict[agent] = critic_target_net_i
        actor_net_dict[agent] = actor_net

    # REPLAY BUFFER
    datamodule = ALGDataModule()
    datamodule.setup(env, actor_net_dict)

    # Create module
    ALG_module_instance = ALGModule(
        ENV,
        critic_net_dict,
        critic_target_net_dict,
        actor_net_dict,
    )

    # Train
    # trainer.fit(model, dm)
    ALG_module_instance.fit(datamodule)

    # Save Results
    if SAVE_RESULTS:
        torch.save(actor_net, 'actor_net.pt')
        # example runs
        model = torch.load('actor_net.pt')
        model.eval()
        play(10, model=model)


if __name__ == '__main__':
    train()


























