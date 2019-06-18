import numpy as np
from torch.autograd import Variable

import config


# agent (or policy) model
class Agent(object):
    def __init__(self, nAction, Q):
        # nAction: number of action (depends on gym environment)
        # Q: policy network

        self.nAction = nAction
        self.Q = Q

    def getAction(self, state, eps):
        # calc best action for given state
        # state: state (4 stacked grayscale frames of size 84x84)
        # eps: value for epsilon greeedy

        var = Variable(state)
        if not config.isLocal:
            var = var.cuda()

        # action with max Q value (q for value and argq for index)
        q, argq = self.Q(var).max(1) 

        # epsilon greedy
        probs = np.full(self.nAction, eps / self.nAction, np.float32)
        probs[argq[0]] += 1 - eps
        return np.random.choice(np.arange(self.nAction), p=probs)
