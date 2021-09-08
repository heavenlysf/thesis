import torch
import torch.nn as nn
import numpy as np


def init_weights(layer, gain):
    for p in layer.parameters():
        # weights
        if len(p.data.shape) >= 2:
            nn.init.orthogonal_(p, gain=gain)
        # bias
        else:
            p.data.zero_()


def all_init_weights(m, gain=2**0.5):
    init_weights(m, gain)


def get_all_connected_edge_index(n):
    edge_index_from = np.repeat(np.arange(n), n)
    edge_index_to = np.tile(np.arange(n), n)
    edge_index = np.stack((edge_index_from, edge_index_to))
    edge_index = torch.from_numpy(edge_index).long()
    return edge_index


def transpose_list(x):
    t = list(zip(*x))
    t = [list(a) for a in t]
    return t


def compute_gae(rewards, values, termination_type, gamma=0.99, lambda_=0.95):
    if termination_type == 1:  # end by reaching goal
        values[-1] = 0
    elif termination_type == 2:  # end by timeout
        pass
    else:
        raise Exception("Wrong termination_type input")
    gae = 0
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
    return advantages


def compute_td_lambda(rewards, values, termination_type, gamma=0.99, lambda_=0.95):
    if termination_type == 1:  # end by reaching goal
        values[-1] = 0
        return_ = 0
    elif termination_type == 2:  # end by timeout
        return_ = values[-1]
    else:
        raise Exception("Wrong termination_type input")
    returns = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + (1-lambda_) * gamma * values[t+1]
        return_ = delta + gamma * lambda_ * return_
        returns.insert(0, return_)
    return returns
