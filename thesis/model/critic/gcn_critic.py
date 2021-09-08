import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.data import Data, Batch

from ..utils import init_weights, all_init_weights, get_all_connected_edge_index


class GCNStack(nn.Module):
    """node_sizes = (concat)input_size + hidden_sizes + output_size"""

    def __init__(self, n_layer, hidden_size, pool_function):
        super(GCNStack, self).__init__()
        self.pool_function = pool_function
        self.gcns = nn.ModuleList()
        for _ in range(n_layer):
            self.gcns.append(GCNConv(hidden_size, hidden_size))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN layers
        for gcn in self.gcns:
            x = F.relu(gcn(x, edge_index))

        # global pooling
        if self.pool_function == 'max':
            x = pyg_nn.global_max_pool(x, batch)
        elif self.pool_function == 'mean':
            x = pyg_nn.global_mean_pool(x, batch)
        elif self.pool_function == 'sum':
            x = pyg_nn.global_add_pool(x, batch)

        return x


class GCNCritic(nn.Module):
    def __init__(self, obs_size, n_agent, n_action, gcn_n_layer, gcn_hidden_size, gcn_pool_function, global_encode_size,
                 local_encode_size, fc1_size, fc2_size):
        super(GCNCritic, self).__init__()
        self.obs_size = obs_size
        self.n_agent = n_agent
        self.n_action = n_action
        self.gcn_hidden_size = gcn_hidden_size

        self.pre_gcn_fc = nn.Linear(obs_size, gcn_hidden_size)
        self.gcns = GCNStack(gcn_n_layer, gcn_hidden_size, gcn_pool_function)
        self.post_gcn_fc = nn.Linear(gcn_hidden_size, global_encode_size)

        self.local_encode_fc = nn.Linear(obs_size, local_encode_size)

        self.fc1 = nn.Linear(global_encode_size + local_encode_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, n_action)

        self.apply(all_init_weights)
        init_weights(self.fc3, gain=1)

    def forward(self, obs_j):
        # input shape: (-1, n_agent, obs_size)
        # output shape: (-1, n_agent, n_action)
        obs = obs_j.view(-1, self.obs_size)

        global_obs = F.relu(self.pre_gcn_fc(obs))

        global_obs = global_obs.view(-1, self.n_agent, self.gcn_hidden_size)
        edge_index = get_all_connected_edge_index(self.n_agent)
        data_list = [Data(x=data, edge_index=edge_index) for data in global_obs]
        batch = Batch.from_data_list(data_list)

        global_obs = self.gcns(batch)
        global_obs = F.relu(self.post_gcn_fc(global_obs))
        local_obs = F.relu(self.local_encode_fc(obs))

        global_obs = global_obs.repeat_interleave(self.n_agent, dim=0)
        x = torch.cat((global_obs, local_obs), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)

        q = q.view(-1, self.n_agent, self.n_action)

        return q
