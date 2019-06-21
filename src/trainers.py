import torch
import torch.nn as nn
from torch.autograd import Variable

import config


# utility class for training Q-Network
class Trainer(object):
    def __init__(self, Q, QTarget, opt, gamma):
        # Q: Q-Network
        # QTarget: Target Q-Network
        # opt: optimizer

        self.Q = Q
        self.QTarget = QTarget
        self.opt = opt

        self.gamma = gamma
        self.lossFunc = nn.SmoothL1Loss()

    def update(self, batch):
        # update model for given batch
        # batch: training batch of (state, action, reward, nextState)

        # extract training batch
        stateBatch = Variable(torch.cat([step.state for step in batch], 0))
        actionBatch = torch.LongTensor([step.action for step in batch])
        rewardBatch = torch.Tensor([step.reward for step in batch])
        nextStateBatch = Variable(torch.cat([step.nextState for step in batch], 0))

        if not config.isLocal:
            stateBatch = stateBatch.cuda()
            actionBatch = actionBatch.cuda()
            rewardBatch = rewardBatch.cuda()
            nextStateBatch = nextStateBatch.cuda()

        # calc values for update model
        qValue = self.Q(stateBatch).gather(1, actionBatch.unsqueeze(1)).squeeze(1)  # Q(s, a)
        qTarget = rewardBatch + self.QTarget(nextStateBatch).detach().max(1)[0] * self.gamma  # r + γmaxQ(s', a')

        L = self.lossFunc(qValue, qTarget)  # loss to equalize Q(s) and r + γmaxQ(s', a')
        self.opt.zero_grad()
        L.backward()
        self.opt.step()  # train for one batch step
