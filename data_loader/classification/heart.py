import numpy as np
import random
import torch
from torch.utils import data


class HeartDataGenerator(data.Dataset):
    def __init__(self, signals, gts, batch_size):
        self.bachsize = batch_size
        # self.signals = signals
        # self.gts = gts
        self.data_num = len(gts)
        shuffled_index = random.sample(range(self.data_num), self.data_num)
        self.signals = [signals[index] for index in shuffled_index]
        self.gts = [gts[index] for index in shuffled_index]
    def __len__(self):
        return int(np.ceil(self.data_num / self.bachsize))

    def __getitem__(self, index):
        # left_bound = index #* self.bachsize
        # right_bound = (index+1) #* self.bachsize
        # if right_bound > self.data_num:
        #     right_bound = self.data_num
        #     left_bound = right_bound - self.bachsize

        # signals = self.signals[left_bound:right_bound]
        # gts = self.gts[left_bound:right_bound]
        signals = self.signals[index]
        gts = self.gts[index]
        signals_out = np.asarray(signals, dtype=np.float32) # .transpose(0, 2, 3, 1)
        gts_out = np.asarray(gts, dtype=np.float32)
        # signals_out = torch.from_numpy(signals_out)
        # gts_out = torch.from_numpy(gts_out)
        return signals_out, gts_out
