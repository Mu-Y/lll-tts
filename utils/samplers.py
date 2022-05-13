import torch
from torch.utils.data.sampler import Sampler, WeightedRandomSampler, SubsetRandomSampler
from dataset.dataset import TextToSpeechDataset
import random


class RandomImbalancedSampler(Sampler):
    """Samples randomly imbalanced dataset (with repetition).

    Argument:
        data_source -- instance of TextToSpeechDataset
    """

    def __init__(self, data_source):

        lebel_freq = {}
        for idx in range(len(data_source)):
            label = data_source.items[idx]['language']
            if label in lebel_freq: lebel_freq[label] += 1
            else: lebel_freq[label] = 1
        self.lebel_freq = lebel_freq

        total = float(sum(lebel_freq.values()))
        weights = [total / lebel_freq[data_source.items[idx]['language']] for idx in range(len(data_source))]

        self._sampler = WeightedRandomSampler(weights, len(weights))

    def __iter__(self):
        return self._sampler.__iter__()

    def __len__(self):
        return len(self._sampler)



class RandomCycleIter:

    def __init__ (self, data, shuffle=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.shuffle = shuffle
        # self.test_mode = test_mode

    def __iter__ (self):
        return self

    def __next__ (self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if self.shuffle:
                random.shuffle(self.data_list)

        return self.data_list[self.i]

class BalancedBatchSampler(Sampler):

    def __init__(self, data_source, batch_size, n_samples, shuffle=True):

        self.label_indices = {}
        for idx in range(len(data_source)):
            label = data_source.items[idx]['language']
            if label not in self.label_indices: self.label_indices[label] = []
            self.label_indices[label].append(idx)
        languages = list(self.label_indices.keys())

        self._samplers = [RandomCycleIter(self.label_indices[i], shuffle) \
                              for i, _ in enumerate(languages)]

        self._batch_size = batch_size
        self.n_samples = n_samples
        self.data_source = data_source

    def __iter__(self):

        batch = []
        iters = [iter(s) for s in self._samplers]
        done = False
        cnt = 0

        while cnt < self.n_samples:
            b = []
            for it in iters:
                idx = next(it)
                b.append(idx)
                cnt += 1
            batch += b
            if len(batch) == self._batch_size:
                yield batch
                batch = []

    def __len__(self):
        return len(self.data_source) // self._batch_size
