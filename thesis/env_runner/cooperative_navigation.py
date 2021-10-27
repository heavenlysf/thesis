import pickle

import supersuit as ss
import torch
from torch.distributions import Categorical

from thesis.pettingzoo.mpe import simple_spread_v2


class CooperativeNavigationRunner:
    def __init__(self, n_agent, n_env, max_step):
        self.n_agent = n_agent
        self.n_env = n_env
        self.max_step = max_step

        self.n_action = 5
        self.obs_size = 4 * n_agent + 2

        # run episodes in parallel
        self.env = simple_spread_v2.parallel_env(N=n_agent, local_ratio=0.5, max_cycles=99999)
        self.env = ss.pettingzoo_env_to_vec_env_v0(self.env)
        self.env = ss.concat_vec_envs_v0(self.env, n_env)

    def rollout(self, actor):
        observations = []
        actions = []
        rewards = []
        probs = []
        entropys = []

        self.env.seed()
        observation_f = self.env.reset()
        for _ in range(self.max_step):
            observation = torch.tensor(observation_f)
            with torch.no_grad():
                policy = Categorical(actor(observation))
            action = policy.sample()
            prob = policy.probs  # probs of all possible actions
            entropy = policy.entropy()

            # env step
            observation_f, reward_f, _, _ = self.env.step(action.numpy())

            # collect reward
            reward = torch.tensor(reward_f[::self.n_agent])

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            probs.append(prob)
            entropys.append(entropy)

        # last step
        last_observation = torch.tensor(observation_f)
        last_policy = Categorical(actor(last_observation))
        last_action = last_policy.sample()

        # list to tensor
        observations = torch.cat(observations)
        actions = torch.cat(actions)
        rewards = torch.stack(rewards)
        probs = torch.cat(probs)
        entropys = torch.cat(entropys)  # only used for logging so shape doesn't matter

        """
        preprocess 

        before
        observations: (max_step, n_env*n_agent, obs_size)   ->  (n_env, max_step, n_agent, obs_size)
        actions: (max_step, n_env*n_agent)                  ->  (n_env, max_step, n_agent)
        rewards: (max_step, n_env)                          ->  (n_env, max_step)
        probs: (max_step, n_env*n_agent, n_action)          ->  (n_env, max_step, n_agent, n_action)
        entropys: (max_step, n_env*n_agent)                     (only used for logging so shape doesn't matter)
        last_observation: (n_env*n_agent, obs_size)         ->  (n_env, n_agent, obs_size)
        last_action: (n_env*n_agent)                        ->  (n_env, n_agent)

        """

        observations = observations.view(self.max_step, self.n_env, self.n_agent, self.obs_size)
        observations = observations.permute(1, 0, 2, 3).contiguous()
        actions = actions.view(self.max_step, self.n_env, self.n_agent)
        actions = actions.permute(1, 0, 2).contiguous()
        rewards = rewards.permute(1, 0).contiguous()
        probs = probs.view(self.max_step, self.n_env, self.n_agent, self.n_action)
        probs = probs.permute(1, 0, 2, 3).contiguous()
        last_observation = last_observation.view(self.n_env, self.n_agent, self.obs_size)
        last_action = last_action.view(self.n_env, self.n_agent)

        return observations, actions, rewards, probs, entropys, last_observation, last_action

    def demo(self, actor, n_env):
        states = []
        probs_out = []
        actions_out = []

        env = simple_spread_v2.parallel_env(N=self.n_agent, local_ratio=0.5, max_cycles=99999)

        for _ in range(n_env):
            observation_j_f = env.reset()
            for _ in range(self.max_step):
                state = env.render(mode='rgb_array')
                observation_j = [obs for obs in observation_j_f.values()]
                policy_j = [Categorical(actor(torch.tensor([obs])).squeeze(0).detach()) for obs in observation_j]
                action_j = [policy.sample() for policy in policy_j]

                prob_out = [policy.probs.tolist() for policy in policy_j]
                action_out = [action.item() for action in action_j]

                # envs step
                action_j_f = {agent: action.item() for agent, action in zip(env.agents, action_j)}
                observation_j_f, reward_j_f, _, _ = env.step(action_j_f)

                states.append(pickle.loads(pickle.dumps(state)))
                probs_out.append(prob_out)
                actions_out.append(action_out)
        env.close()

        return states, probs_out, actions_out
