import torch.nn as nn
import torch.nn.functional as F


# Q-Network model
class QNet(nn.Module):
    def __init__(self, nAction):
        # nAction: number of action (depends on gym environment)

        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # (4, 84, 84) -> (32, 20, 20)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # (32, 20, 20) -> (64, 9, 9)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # (64, 9, 9) -> (64, 7, 7)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, nAction)

    def forward(self, x):
        # run forward propagation
        # x: state (4 stacked grayscale frames of size 84x84)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))  # flatten
        return self.fc2(x)
