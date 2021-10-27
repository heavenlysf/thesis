import time

import numpy as np
import torch
import wandb

from thesis.factory import env_factory, critic_factory
from thesis.logger import Logger
from thesis.model import ActorMlp
from thesis.utils import compute_td_lambda


class Coma:
    def __init__(self, env_name, critic_type, n_env, max_step, gamma, lambda_, actor_lr, critic_lr,
                 grad_norm_clip, actor_fc_size, log_to_wandb, env_kwarg, critic_kwarg):
        self.n_env = n_env
        self.max_step = max_step
        self.gamma = gamma
        self.lambda_ = lambda_
        self.grad_norm_clip = grad_norm_clip
        self.log_to_wandb = log_to_wandb

        self.env_runner = env_factory(env_name, **env_kwarg)
        self.obs_size = self.env_runner.obs_size
        self.n_agent = self.env_runner.n_agent
        self.n_action = self.env_runner.n_action
        self.actor = ActorMlp(self.obs_size, self.n_action, actor_fc_size)
        self.critic = critic_factory(critic_type, **critic_kwarg)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.logger = Logger()

    def run(self, iterations):
        for iteration in range(iterations):
            # rollout
            start_iter = time.time()
            (observations, actions, rewards, probs,
             entropys, last_observation, last_action) = self.env_runner.rollout(self.actor)

            # training
            start_train = time.time()
            (log_reward, log_q_taken, log_critic_loss, log_actor_loss,
             log_actor_loss_amount, log_entropy, log_residual_variance) = self._train(observations, actions, rewards,
                                                                                      probs, entropys, last_observation,
                                                                                      last_action)

            # logging
            start_log = time.time()
            self.logger.append(log_reward, log_q_taken, log_critic_loss, log_actor_loss, log_actor_loss_amount,
                               log_entropy, log_residual_variance)
            if self.log_to_wandb:
                wandb.log({'reward': log_reward, 'q_taken': log_q_taken, 'critic_loss': log_critic_loss,
                           'actor_loss': log_actor_loss, 'entropy': log_entropy,
                           'residual_variance': log_residual_variance})

            end_iter = time.time()
            print("iteration =", iteration + 1,
                  "\ttime elapsed =", "{:.4f}".format(end_iter - start_iter),
                  "\trollout time =", "{:.4f}".format(start_train - start_iter),
                  "\ttraining time =", "{:.4f}".format(start_log - start_train),
                  "\tlogging time =", "{:.4f}".format(end_iter - start_log))

    def _train(self, observations, actions, rewards, probs, entropys, last_observation, last_action):
        # get values
        with torch.no_grad():
            q_all = self.critic(observations.view(-1, self.n_agent, self.obs_size))
            last_q_all = self.critic(last_observation)

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
        actions_taken = actions.view(-1, 1).long()
        last_action_taken = last_action.view(-1, 1).long()

        values = torch.gather(q_all, dim=1, index=actions_taken)
        last_value = torch.gather(last_q_all, dim=1, index=last_action_taken)

        values = values.view(self.n_agent, self.n_env, self.max_step)
        last_value = last_value.view(self.n_agent, self.n_env, 1)

        # td lambda
        returns = []
        for i in range(self.n_agent):
            returns_agent = []
            for j in range(self.n_env):
                rewards_ep = rewards[j]
                values_ep = torch.cat((values[i][j], last_value[i][j]))
                returns_ep = compute_td_lambda(rewards_ep, values_ep, 2, self.gamma, self.lambda_)
                returns_agent.append(torch.tensor(returns_ep))
            returns.append(torch.stack(returns_agent))
        returns = torch.stack(returns)
        returns = returns.permute(2, 1, 0).contiguous()  # returns: (max_step, n_env, n_agent)

        # reshape to (max_step, n_env, n_agent, -1) because coma algo requires it.
        # I don't like how each minibatch is a single timestep btw.
        observations = observations.permute(1, 0, 2, 3).contiguous()
        actions = actions.permute(1, 0, 2).contiguous()
        probs = probs.permute(1, 0, 2, 3).contiguous()

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
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm_clip, norm_type=2)
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
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm_clip, norm_type=2)
        self.actor_optimizer.step()

        # calculate residual variance
        log_actor_loss_amount = (log_probs_taken * advantages).detach().abs().mean().item()
        residual_variance_log_1 = torch.cat(residual_variance_log_1)
        residual_variance_log_2 = torch.cat(residual_variance_log_2)
        residual_variance = torch.var(residual_variance_log_1) / torch.var(residual_variance_log_2)

        # collect logs
        log_reward = rewards.mean().item()
        log_q_taken = q_taken.mean().item()
        log_critic_loss = critic_loss_
        log_actor_loss = actor_loss.detach().item()
        log_actor_loss_amount = log_actor_loss_amount
        log_entropy = entropys.mean().item() / np.log(self.n_action)
        log_residual_variance = residual_variance.item()

        return (log_reward, log_q_taken, log_critic_loss, log_actor_loss, log_actor_loss_amount,
                log_entropy, log_residual_variance)

    def demo(self, n_env):
        states, probs_out, actions_out = self.env_runner.demo(self.actor, n_env)

        return states, probs_out, actions_out

    def get_log(self):
        return self.logger.get_log()

    def load_model(self, actor_state, critic_state):
        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(critic_state)

    def get_model(self):
        return self.actor.state_dict(), self.critic.state_dict()
