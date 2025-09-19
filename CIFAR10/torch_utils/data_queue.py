import torch


class Data_queue(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.init()

    def init(self):
        self.bank = self.create_list_with_capacity(self.capacity, 0)
        self.bank_ptr = 0

    def create_list_with_capacity(self, capacity, default_value=None):
        return [default_value] * capacity

    def set_data(self, data):
        ptr = self.bank_ptr
        if ptr >= self.capacity:
            self.bank[0] = data
            self.bank_ptr = 0
        else:
            self.bank[ptr] = data
        self.bank_ptr += 1

    def get_data(self):
        return self.bank


