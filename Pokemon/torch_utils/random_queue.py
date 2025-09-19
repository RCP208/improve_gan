# https://github.com/taufikxu/GAN_PID/blob/master/gan_training/random_queue.py
import torch
import numpy as np


class Random_queue_torch(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.length = 0
        self.data = None
        self.label = None
        self.init([3, 32, 32])
        self.FLAG = False

    def init(self, dim):
        self.bank = torch.randn([self.capacity, *dim])
        self.bank_ptr = 0

    def set_data(self, batch: torch.Tensor):
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """
        batch_size = batch.shape[0]
        ptr = self.bank_ptr

        if ptr + batch_size >= self.capacity:
            self.bank[ptr:] = batch[:self.capacity - ptr].detach()
            self.bank_ptr = 0
            self.FLAG = True

        else:
            self.bank[ptr:ptr + batch_size] = batch.detach()
            self.bank_ptr = ptr + batch_size

    def get_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if not self.FLAG and batch_size > self.bank_ptr:
            return self.bank[:self.bank_ptr]
        # results = []
        if self.FLAG:
            permutation = np.random.permutation(self.capacity)
        else:
            permutation = np.random.permutation(self.bank_ptr)

        result = self.bank[permutation[:batch_size]]
        return result

    def is_empty(self):
        return self.length == 0