from alg_constrants_amd_packages import *
from alg_general_functions import *
from alg_nets import *
from alg_data_module import *
from alg_module import *
from alg_env_module import env_module


def train():
    # Initialization
    # NETS
    critic_net_dict, critic_target_net_dict, actor_net_dict, actor_target_net_dict = {}, {}, {}, {}
    for agent in env_module.get_agent_list():

        obs_size, n_actions = env_module.observation_space_shape(agent), env_module.action_space_shape(agent)
        n_agents = len(env_module.get_agent_list())

        critic_net_i = CriticNet(obs_size, n_actions, n_agents)
        critic_target_net_i = CriticNet(obs_size, n_actions, n_agents)
        critic_target_net_i.load_state_dict(critic_net_i.state_dict())
        actor_net_i = ActorNet(obs_size, n_actions)
        actor_target_net_i = ActorNet(obs_size, n_actions)
        actor_target_net_i.load_state_dict(actor_net_i.state_dict())

        critic_net_dict[agent] = critic_net_i
        critic_target_net_dict[agent] = critic_target_net_i
        actor_net_dict[agent] = actor_net_i
        actor_target_net_dict[agent] = actor_target_net_i

    # REPLAY BUFFER
    datamodule = ALGDataModule()
    datamodule.setup(actor_net_dict)

    # Create module
    ALG_module_instance = ALGModule(
        env_module.env,
        critic_net_dict,
        critic_target_net_dict,
        actor_net_dict,
        actor_target_net_dict,
    )

    # Train
    # trainer.fit(model, dm)
    ALG_module_instance.fit(datamodule)

    # Save Results
    if SAVE_RESULTS:
        torch.save(list(actor_net_dict.values())[0], 'data/actor_net.pt')
        # example runs
        model = torch.load('actor_net.pt')
        model.eval()
        models_dict = {agent: model for agent in env_module.get_agent_list()}
        play(10, models_dict=models_dict)


if __name__ == '__main__':
    train()


























