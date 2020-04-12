import torch.nn as nn
import torch.nn.functional as F

from . import utils

# linear model
class MyLinearNet(utils.ReparamModule):
    def __init__(self, state):
        super(MyLinearNet, self).__init__()
        self.fc = nn.Linear(2, 1 if state.num_classes <= 2 else state.num_classes)
        self.l2 = state.L2_coef
#         self.fc2 = nn.Linear(20, )

    def forward(self, x):
#         out = F.relu(self.fc1(x), inplace=True)
        out = self.fc(x)
        if self.training:
            for p in self.parameters():
                out = out + self.l2*(p**2).sum()
        return out


class MyNonLinearNet(utils.ReparamModule):
    def __init__(self, state, mid_sz = 10):
        # print(type(MyNonLinearNet), type(self))
        super(MyNonLinearNet, self).__init__()
        self.fc1 = nn.Linear(2, mid_sz)
        self.fc2 = nn.Linear(mid_sz, 1 if state.num_classes <= 2 else state.num_classes)

    def forward(self, x):
        out = F.relu(self.fc1(x), inplace=True)
        out = self.fc2(out)
        return out


class MyMoreNonLinearNet(utils.ReparamModule):
    def __init__(self, state, mid_sz = 10):
        # print(type(MyNonLinearNet), type(self))
        super(MyMoreNonLinearNet, self).__init__()
        self.fc1 = nn.Linear(2, mid_sz)
        self.fc2 = nn.Linear(mid_sz, mid_sz if state.num_classes <= 2 else state.num_classes)
        self.fc3 = nn.Linear(mid_sz, mid_sz if state.num_classes <= 2 else state.num_classes)
        self.fc4 = nn.Linear(mid_sz, 1 if state.num_classes <= 2 else state.num_classes)

    def forward(self, x):
        out = F.relu(self.fc1(x), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = F.relu(self.fc3(out), inplace=True)
        out = self.fc4(out)
        return out
