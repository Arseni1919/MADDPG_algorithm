from alg_constrants_amd_packages import *
from alg_general_functions import *
from alg_plotter import plotter

class ALGDataset(Dataset):
    def __init__(self):
        self.buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, indx):
        item = self.buffer[indx]
        return item.state, item.action, item.reward, item.done, item.new_state

    def append(self, experience):
        self.buffer.append(experience)


class ALGDataModule:
    def __init__(self, data_dir: str = "/data", batch_size: int = 32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dataset = ALGDataset()
        self.filled_the_dataset = False
        plotter.info("ALGDataModule instance created.")

    def prepare_data(self):
        # download
        pass

    def setup(self, actor_net_dict):
        self.fill_the_buffer(actor_net_dict)

    def train_dataloader(self):
        if not self.filled_the_dataset:
            plotter.error("Didn't fill the dataset.")
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def fill_the_buffer(self,  actor_net_dict):
        while len(self.train_dataset) < UPDATE_AFTER:
            for step in env_module.run_episode(models_dict=actor_net_dict):
                experience, observations, actions, rewards, dones, new_observations = step
                self.train_dataset.append(experience)
        plotter.info("Filled the dataset.")
        self.filled_the_dataset = True

















        # with torch.no_grad():
        #     observations = env.reset()
        #     dones = {agent: False for agent in env.agents}
        #     while len(self.train_dataset) < UPDATE_AFTER:
        #         actions = get_actions(env, observations, dones, actor_net_dict)
        #         new_observations, rewards, dones, infos = env.step(actions)
        #
        #         # ADD TO EXPERIENCE BUFFER
        #         experience = Experience(state=observations, action=actions, reward=rewards, done=dones,
        #                                 new_state=new_observations)
        #         self.train_dataset.append(experience)
        #         observations = new_observations
        #
        #         if all(dones.values()):
        #             observations = self.env.reset()
        #             dones = {agent: False for agent in self.env.agents}
        #
        #     env.close()
        #     self.filled_the_dataset = True












