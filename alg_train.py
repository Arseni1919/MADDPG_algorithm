from alg_constrants_amd_packages import *
from alg_general_functions import *
from alg_logger import run
from alg_nets import *
from alg_memory import *
from alg_module import *


def train():
    # Initialization

    # ENV
    env = gym.make(ENV)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    # NETS
    critic_net_1 = CriticNet(obs_size, n_actions)
    critic_target_net_1 = CriticNet(obs_size, n_actions)
    critic_target_net_1.load_state_dict(critic_net_1.state_dict())

    critic_net_2 = CriticNet(obs_size, n_actions)
    critic_target_net_2 = CriticNet(obs_size, n_actions)
    critic_target_net_2.load_state_dict(critic_net_2.state_dict())

    actor_net = ActorNet(obs_size, n_actions)

    # REPLAY BUFFER
    train_dataset = ALGDataset()
    fill_the_buffer(train_dataset, env, actor_net)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Train
    ALG_module_instance = ALGModule(
        env,
        critic_net_1,
        critic_target_net_1,
        critic_net_2,
        critic_target_net_2,
        actor_net,
        train_dataset,
        train_dataloader
    )
    ALG_module_instance.fit()

    # Save Results
    if SAVE_RESULTS:
        torch.save(actor_net, 'actor_net.pt')
        # example runs
        model = torch.load('actor_net.pt')
        model.eval()
        play(10, model=model)


if __name__ == '__main__':
    train()


























