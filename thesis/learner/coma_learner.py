from .base import ILearner
from .utils import compute_td_lambda

import torch
import numpy as np


class ComaLearner(ILearner):
    def __init__(self, logger, actor, critic, actor_optimizer, critic_optimizer,
                 n_agent, n_env, max_step, n_action, obs_size,
                 observations, actions, rewards, probs, entropys, last_observation, last_action,
                 gamma, lambda_):
        self.logger = logger
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.n_agent = n_agent
        self.n_env = n_env
        self.max_step = max_step
        self.n_action = n_action
        self.obs_size = obs_size
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.probs = probs
        self.entropys = entropys
        self.last_observation = last_observation
        self.last_action = last_action
        self.gamma = gamma
        self.lambda_ = lambda_

    def train(self):
        # get values
        with torch.no_grad():
            q_all = self.critic(self.observations.view(-1, self.n_agent, self.obs_size))
            last_q_all = self.critic(self.last_observation)

        # torch.gather() shenanigans
        # prepare variables for returns calculation by shaping them to (n_agent, n_env, max_step)
        '''
        q_all : (n_env*max_step, n_agent, n_action) -> (n_env*max_step*n_agent, n_action)
        last_q_all: (n_env, n_agent, n_action) -> (n_env*n_agent, n_action)
        '''
        q_all = q_all.view(-1, self.n_action)
        last_q_all = last_q_all.view(-1, self.n_action)

        '''
        actions_taken: (n_agent*n_env*max_step, 1)
        last_action_taken: (n_agent*n_env, 1)
        '''
        actions_taken = self.actions.view(-1, 1).long()
        last_action_taken = self.last_action.view(-1, 1).long()

        values = torch.gather(q_all, dim=1, index=actions_taken)
        last_value = torch.gather(last_q_all, dim=1, index=last_action_taken)

        values = values.view(self.n_agent, self.n_env, self.max_step)
        last_value = last_value.view(self.n_agent, self.n_env, 1)

        # td lambda
        returns = []
        for i in range(self.n_agent):
            returns_agent = []
            for j in range(self.n_env):
                rewards_ep = self.rewards[j]
                values_ep = torch.cat((values[i][j], last_value[i][j]))
                returns_ep = compute_td_lambda(rewards_ep, values_ep, 2, self.gamma, self.lambda_)
                returns_agent.append(torch.tensor(returns_ep))
            returns.append(torch.stack(returns_agent))
        returns = torch.stack(returns)
        returns = returns.permute(2, 1, 0).contiguous()  # returns: (max_step, n_env, n_agent)

        # reshape to (max_step, n_env, n_agent, -1) because coma algo requires it.
        # I don't like how each minibatch is a single timestep btw.
        observations = self.observations.permute(1, 0, 2, 3).contiguous()
        actions = self.actions.permute(1, 0, 2).contiguous()
        probs = self.probs.permute(1, 0, 2, 3).contiguous()

        # update critic
        log_critic_loss = []
        residual_variance_log_1 = []
        residual_variance_log_2 = []
        q_values = []
        for t in reversed(range(self.max_step)):
            q_t = self.critic(observations[t])
            action_taken_t = actions[t].view(-1, 1).long()
            q_t_taken = torch.gather(q_t.view(-1, self.n_action), dim=1, index=action_taken_t).view(-1)
            return_t = returns[t].view(-1)
            critic_loss = (q_t_taken - return_t).pow(2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0, norm_type=2)
            self.critic_optimizer.step()

            residual_variance_log_1.append((return_t - q_t_taken.detach()))
            residual_variance_log_2.append(return_t)

            q_values.insert(0, q_t.detach())
            log_critic_loss.append(critic_loss.detach())
        q_values = torch.stack(q_values, 0)
        critic_loss_ = torch.tensor(log_critic_loss).mean().item()

        # baseline
        baseline_values = (probs * q_values).sum(3).view(-1)
        actions_taken = actions.view(-1, 1).long()
        q_taken = torch.gather(q_values.view(-1, self.n_action), dim=1, index=actions_taken).view(-1)
        advantages = q_taken - baseline_values

        # update actor
        policies = self.actor(observations.view(-1, self.obs_size)).view(-1, self.n_action)
        probs_taken = torch.gather(policies, dim=1, index=actions_taken).view(-1)
        log_probs_taken = torch.log(probs_taken)

        actor_loss = -(log_probs_taken * advantages).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0, norm_type=2)
        self.actor_optimizer.step()

        # calculate residual variance
        log_actor_loss_amount = (log_probs_taken * advantages).detach().abs().mean().item()
        residual_variance_log_1 = torch.cat(residual_variance_log_1)
        residual_variance_log_2 = torch.cat(residual_variance_log_2)
        residual_variance = torch.var(residual_variance_log_1) / torch.var(residual_variance_log_2)

        # collect logs
        log_reward = self.rewards.mean().item()
        log_q_taken = q_taken.mean().item()
        log_critic_loss = critic_loss_
        log_actor_loss = actor_loss.detach().item()
        log_actor_loss_amount = log_actor_loss_amount
        log_entropy = self.entropys.mean().item()/np.log(self.n_action)
        log_residual_variance = residual_variance.item()

        return (log_reward, log_q_taken, log_critic_loss, log_actor_loss, log_actor_loss_amount,
                log_entropy, log_residual_variance)
