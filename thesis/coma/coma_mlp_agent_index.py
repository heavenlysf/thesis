from base import IComa
from ..learner.coma_learner import ComaLearner
from ..runner.cooperative_navigation_runner_mlp import MLPCooperativeNavigationRunner
from ..model.actor.mlp_actor import MLPActor
from ..model.critic.mlp_agent_index_critic import MLPAgentIndexCritic
from thesis.pettingzoo.mpe import simple_spread_v2
from thesis.logger import Logger

import time
import torch
import torch.optim as optim
from torch.distributions import Categorical
import pickle
import wandb


class ComaMLPAgentIndex(IComa):
    def __init__(self, n_agent, n_env, max_step, gamma, lambda_,
                 actor_hidden_size, fc1_size, fc2_size,
                 actor_lr, critic_lr, log_to_wandb=False):
        self.n_agent = n_agent
        self.n_env = n_env
        self.max_step = max_step
        self.gamma = gamma
        self.lambda_ = lambda_
        self.log_to_wandb = log_to_wandb

        self.n_action = 5
        self.obs_size = 4 * n_agent + 2

        self.actor = MLPActor(self.obs_size, self.n_action, actor_hidden_size)
        self.critic = MLPAgentIndexCritic(self.obs_size, self.n_agent, self.n_action, fc1_size, fc2_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.logger = Logger()

    def run(self, iterations):
        for iteration in range(iterations):
            # rollout
            start_iter = time.time()
            runner = MLPCooperativeNavigationRunner(self.actor, self.n_agent, self.n_env, self.max_step)
            observations, actions, rewards, probs, entropys, last_observation, last_action = runner.rollout()

            # training
            start_train = time.time()
            learner = ComaLearner(self.logger, self.actor, self.critic, self.actor_optimizer,
                                  self.critic_optimizer, self.n_agent, self.n_env, self.max_step,
                                  self.n_action, self.obs_size, observations, actions, rewards, probs,
                                  entropys, last_observation, last_action, self.gamma, self.lambda_)
            (log_reward, log_q_taken, log_critic_loss, log_actor_loss, log_actor_loss_amount,
             log_entropy, log_residual_variance) = learner.train()

            # logging
            start_log = time.time()
            self.logger.append(log_reward, log_q_taken, log_critic_loss, log_actor_loss, log_actor_loss_amount,
                               log_entropy, log_residual_variance)
            # log to wandb
            if self.log_to_wandb:
                wandb.log({'reward': log_reward, 'q_taken': log_q_taken, 'critic_loss': log_critic_loss,
                           'actor_loss': log_actor_loss, 'entropy': log_entropy,
                           'residual_variance': log_residual_variance})

            end_iter = time.time()
            print("iteration =", iteration+1,
                  "\ttime elapsed =", "{:.4f}".format(end_iter-start_iter),
                  "\trollout time =", "{:.4f}".format(start_train-start_iter),
                  "\ttraining time =", "{:.4f}".format(start_log-start_train),
                  "\tlogging time =", "{:.4f}".format(end_iter-start_log))

    def demo(self, n_env):
        states = []
        probs_out = []
        actions_out = []

        env = simple_spread_v2.parallel_env(N=self.n_agent, local_ratio=0.5, max_cycles=99999)

        for _ in range(n_env):
            observation_j_f = env.reset()
            for _ in range(self.max_step):
                state = env.render(mode='rgb_array')
                observation_j = [obs for obs in observation_j_f.values()]
                policy_j = [Categorical(self.actor(torch.tensor([obs])).squeeze(0).detach()) for obs in observation_j]
                action_j = [policy.sample() for policy in policy_j]

                prob_out = [policy.probs.tolist() for policy in policy_j]
                action_out = [action.item() for action in action_j]

                # env step
                action_j_f = {agent: action.item() for agent, action in zip(env.agents, action_j)}
                observation_j_f, reward_j_f, _, _ = env.step(action_j_f)

                states.append(pickle.loads(pickle.dumps(state)))
                probs_out.append(prob_out)
                actions_out.append(action_out)
        env.close()

        return states, probs_out, actions_out

    def get_log(self):
        return self.logger.get_log()

    def load_network(self, actor_state, critic_state):
        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(critic_state)

    def get_network(self):
        return self.actor.state_dict(), self.critic.state_dict()
