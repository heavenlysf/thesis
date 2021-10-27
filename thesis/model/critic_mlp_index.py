import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.utils import init_weights, all_init_weights


class CriticMlpIndex(nn.Module):
    def __init__(self, obs_size, n_agent, n_action, fc1_size, fc2_size):
        super(CriticMlpIndex, self).__init__()
        self.obs_size = obs_size
        self.n_agent = n_agent
        self.n_action = n_action

        self.fc1 = nn.Linear(obs_size * n_agent + n_agent, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, n_action)

        self.apply(all_init_weights)
        init_weights(self.fc3, gain=1)

    def forward(self, obs_j):
        # input shape: (-1, n_agent, obs_size)
        # output shape: (-1, n_agent, n_action)
        one_hot_indices = torch.Tensor(np.tile(np.identity(self.n_agent), reps=(obs_j.size(0), 1)))
        obs_j = obs_j.contiguous().view(obs_j.size(0), -1)
        obs_j = obs_j.repeat_interleave(self.n_agent, dim=0)
        x = torch.cat((obs_j, one_hot_indices), 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)

        q = q.view(-1, self.n_agent, self.n_action)

        return q
