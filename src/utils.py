import random
from typing import NamedTuple

import torch
import torchvision.transforms as T


# one step of interaction with environment
class Step(NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    nextState: torch.Tensor


# replay buffer
class ReplayBuffer(object):
    def __init__(self, capacity):
        # capacity: max size of replay buffer

        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, step):
        # add  a step to buffer
        # step: one step of interaction

        if len(self.memory) < self.capacity:
            self.memory.append(step)
        else:
            self.memory[self.index] = step

        self.index = (self.index + 1) % self.capacity

    def sample(self, size):
        # collect batch of given size
        # size: batch size

        return random.sample(self.memory, size)


def preprocess(x):
    # preprocess frame
    # x: a frame of size 210x160

    # resize, grayscale and convert to tensor
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(84),
        T.Grayscale(),
        T.ToTensor()
    ])

    return transform(x[50:, :, :]).unsqueeze(0)
