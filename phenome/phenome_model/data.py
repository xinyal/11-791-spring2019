from __future__ import print_function

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


class TrainDataloader(DataLoader):
    def __init__(self, data, label, batch_size):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.bucket_sizes = [300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 2000]
        self.buckets = [[] for _ in self.bucket_sizes]
        for idx, datapoint in enumerate(data):
            for b_id, b_size in enumerate(self.bucket_sizes):
                if datapoint.shape[0] <= b_size:
                    self.buckets[b_id].append(idx)
                    break
        bucket_lengths = [len(bucket) for bucket in self.buckets]
        print(bucket_lengths, sum(bucket_lengths))

        self.batches = []
        for b_id, bucket in enumerate(self.buckets):
            bz_size = self.batch_size * 800 // self.bucket_sizes[b_id]
            print(bz_size)
            start_index = 0
            end_index = start_index + bz_size
            while end_index < len(bucket):
                self.batches.append((b_id, start_index, end_index))
                start_index = end_index
                end_index = start_index + bz_size
            self.batches.append((b_id, start_index, len(bucket)))

        # print(self.batches)
        print(len(self.batches))
        
    def __iter__(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            bucket, start, end = batch
            selected_datapoint = np.zeros((end - start), dtype=int)
            for idx in range(end - start):
                selected_datapoint[idx] = self.buckets[bucket][idx + start]
            yield self.data[selected_datapoint], self.label[selected_datapoint]


def load_train():
    data = np.load('../data/wsj0_train.npy', encoding='latin1')
    label = np.load('../data/wsj0_train_merged_labels.npy', encoding='latin1')
    print(data.shape)
    print(label.shape)
    
    loader = TrainDataloader(data, label, 32)
    
    return loader

def load_dev():
    data = np.load('../data/wsj0_dev.npy', encoding='latin1')
    label = np.load('../data/wsj0_dev_merged_labels.npy', encoding='latin1')
    print(data.shape)
    print(label.shape)

    loader = TrainDataloader(data, label, 32)

    return loader


class TestDataLoader(DataLoader):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        num_batch = (self.data.shape[0] + self.batch_size - 1) // self.batch_size
        for b in range(num_batch):
            start = b * self.batch_size
            end = min(start + self.batch_size, self.data.shape[0])
            yield self.data[start : end]


def load_test():
    data = np.load('../data/wsj0_test.npy', encoding='latin1')
    print(data.shape)

    loader = TestDataLoader(data, 16)
    return loader


if __name__ == '__main__':
    load_train()