from alg_GLOBALS import *
from alg_nets import *
from alg_plotter import ALGPlotter
from alg_env_wrapper import MultiAgentParallelEnvWrapper


def train():
    pass


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
    # ENV = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=MAX_CYCLES, continuous_actions=True)
    ENV_NAME = 'simple_v2'
    ENV = simple_v2.parallel_env(max_cycles=25, continuous_actions=True)
    env = MultiAgentParallelEnvWrapper(ENV)

    NUMBER_OF_GAMES = 10

    # --------------------------- # FOR PLOT # -------------------------- #
    PLOT_PER = 1
    NEPTUNE = False
    PLOT_LIVE = True
    SAVE_RESULTS = True
    SAVE_PATH = f'data/actor_{ENV_NAME}.pt'
    plotter = ALGPlotter(plot_life=PLOT_LIVE, plot_neptune=NEPTUNE, name='my_run_ppo', tags=[ENV_NAME])

    # --------------------------- # NETS # -------------------------- #
    critic = CriticNet(obs_size=env.observation_size(env.agents[0]), n_actions=env.action_size(env.agents[0]))
    actor = ActorNet(obs_size=env.observation_size(env.agents[0]), n_actions=env.action_size(env.agents[0]))
    actor_old = ActorNet(obs_size=env.observation_size(env.agents[0]), n_actions=env.action_size(env.agents[0]))
    # --------------------------- # OPTIMIZERS # -------------------------- #
    critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)

    # --------------------------- # REPLAY BUFFER # -------------------------- #
    # replay_buffer = ReplayBuffer()

    # --------------------------- # PLOTTER INIT # -------------------------- #

    # --------------------------- # SEED # -------------------------- #

    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    # MAIN PROCESS
    train()

    # Example Plays
    print(colored('Example run...', 'green'))
    # TODO
    # load_and_play(parallel_env, 5, path_to_save)


# print(colored(f'\n~[WARNING]: {message}', 'yellow'), end=end)


























