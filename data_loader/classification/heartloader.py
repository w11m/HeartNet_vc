import numpy as np
import random
import torch
from torch.utils import data


class HeartGenerator2(data.Dataset):
    def __init__(self, signals, gts, batch_size):
        self.bachsize = batch_size
        self.data_num = len(gts)
        shuffled_index = random.sample(range(self.data_num), self.data_num)
        self.signals = [signals[index] for index in shuffled_index]
        self.gts = [gts[index] for index in shuffled_index]
    def __len__(self):
        return int(np.ceil(self.data_num / self.bachsize))

    def __getitem__(self, index):
        signals = self.signals[index]
        gts = self.gts[index]
        signals_out = np.asarray(signals, dtype=np.float32) # .transpose(0, 2, 3, 1)
        gts_out = np.asarray(gts, dtype=np.float32)
        return signals_out, gts_out