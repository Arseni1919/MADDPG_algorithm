from alg_constrants_amd_packages import *


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

