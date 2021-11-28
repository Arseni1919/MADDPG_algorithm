from alg_GLOBALS import *
from alg_nets import *
from alg_plotter import ALGPlotter


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
        torch.save(list(actor_net_dict.values())[0], SAVE_PATH)
        # example runs
        model = torch.load(SAVE_PATH)
        model.eval()
        models_dict = {agent: model for agent in env_module.get_agent_list()}
        play(10, models_dict=models_dict)


if __name__ == '__main__':

    # --------------------------- # PARAMETERS # -------------------------- #
    M_EPISODES = 1000
    BATCH_SIZE = 64  # size of the batches
    REPLAY_BUFFER_SIZE = BATCH_SIZE * 1000
    LR_CRITIC = 1e-2  # learning rate
    LR_ACTOR = 1e-2  # learning rate
    GAMMA = 0.95  # discount factor
    ACT_NOISE = 0.5  # actuator noise
    POLYAK = 0.99
    VAL_EVERY = 2000
    TRAIN_EVERY = 100

    # --------------------------- # CREATE ENV # -------------------------- #
    NUMBER_OF_AGENTS = 1
    # ENV = simple_spread_v2.env(N=3, local_ratio=0.5, max_cycles=MAX_CYCLES, continuous_actions=True)
    ENV_NAME = 'simple_v2'
    ENV = simple_v2.env(max_cycles=25, continuous_actions=True)
    env = MultiAgentEnv(env=ENV)

    NUMBER_OF_GAMES = 10

    # --------------------------- # FOR PLOT # -------------------------- #
    PLOT_PER = 1
    NEPTUNE = False
    PLOT_LIVE = True
    SAVE_RESULTS = True
    SAVE_PATH = f'data/actor_{ENV_NAME}.pt'
    plotter = ALGPlotter(plot_life=PLOT_LIVE, plot_neptune=NEPTUNE, name='my_run_ppo', tags=[ENV_NAME])

    # --------------------------- # NETS # -------------------------- #
    critic = CriticNet(obs_size=env.observation_size(), n_actions=env.action_size())
    actor = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
    actor_old = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
    # --------------------------- # OPTIMIZERS # -------------------------- #
    critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)

    # --------------------------- # REPLAY BUFFER # -------------------------- #
    # replay_buffer = ReplayBuffer()

    # --------------------------- # PLOTTER INIT # -------------------------- #

    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    # MAIN PROCESS
    train()

    # Example Plays
    print(colored('Example run...', 'green'))
    # TODO
    # load_and_play(env, 5, path_to_save)


# print(colored(f'\n~[WARNING]: {message}', 'yellow'), end=end)


























