from ..utils import init_weights, all_init_weights

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPCritic(nn.Module):
    def __init__(self, obs_size, n_agent, n_action, global_encode_size, local_encode_size, fc1_size, fc2_size):
        super(MLPCritic, self).__init__()
        self.obs_size = obs_size
        self.n_agent = n_agent
        self.n_action = n_action

        self.global_encode_fc1 = nn.Linear(obs_size * n_agent, global_encode_size)
        self.global_encode_fc2 = nn.Linear(global_encode_size, global_encode_size)
        self.local_encode_fc = nn.Linear(obs_size, local_encode_size)

        self.fc1 = nn.Linear(global_encode_size + local_encode_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, n_action)

        self.apply(all_init_weights)
        init_weights(self.fc3, gain=1)

    def forward(self, obs_j):
        # input shape: (-1, n_agent, obs_size)
        # output shape: (-1, n_agent, n_action)
        global_obs = obs_j.view(-1, self.obs_size * self.n_agent)
        global_obs = F.relu(self.global_encode_fc1(global_obs))
        global_obs = F.relu(self.global_encode_fc2(global_obs))

        local_obs = obs_j.view(-1, self.obs_size)
        local_obs = F.relu(self.local_encode_fc(local_obs))

        global_obs = global_obs.repeat_interleave(self.n_agent, dim=0)
        x = torch.cat((global_obs, local_obs), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)

        q = q.view(-1, self.n_agent, self.n_action)

        return q
