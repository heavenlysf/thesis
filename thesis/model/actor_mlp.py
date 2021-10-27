import torch.nn as nn
import torch.nn.functional as F

from thesis.utils import init_weights, all_init_weights


class ActorMlp(nn.Module):
    def __init__(self, obs_size, n_action, hidden_size):
        super(ActorMlp, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_action)

        self.apply(all_init_weights)
        init_weights(self.fc3, gain=0.01)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        dist = F.softmax(self.fc3(x), dim=1)

        return dist
