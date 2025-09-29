import numpy as np

from rob831.agents.base_agent import BaseAgent
from rob831.policies.MLP_policy import MLPPolicyPG
from rob831.infrastructure.replay_buffer import ReplayBuffer

from rob831.infrastructure.utils import normalize, unnormalize

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super().__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        q_vals = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(
            observations, rewards_list, q_vals, terminals
        )
        train_log = self.actor.update(
            observations, actions, advantages, q_values=q_vals
        )
        return train_log


    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        q_values = []
        
        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        if not self.reward_to_go:
            #use the whole traj for each timestep
            for rewards in rewards_list:
                q_values.append(self._discounted_return(rewards))

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            for rewards in rewards_list:
                q_values.append(self._discounted_cumsum(rewards))

        return q_values  # return an array

    def estimate_advantage(self, obs, rewards_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:

            values_normalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            q_values = np.asarray(q_values)
            assert values_normalized.ndim == q_values.ndim
            ## TODO: values were trained with standardized q_values, so ensure
                ## that the predictions have the same mean and standard deviation as
                ## the current batch of q_values

            q_values_array = np.asarray(q_values)
            q_mean = np.mean(q_values_array)
            q_std = np.std(q_values_array)
            if q_std < 1e-8:
                values = np.full_like(values_normalized, q_mean)
            else:
                values = unnormalize(values_normalized, q_mean, q_std)

            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rewards = np.concatenate(rewards_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    ## TODO: recursively compute advantage estimates starting from
                        ## timestep T.
                    ## HINT 1: use terminals to handle edge cases. terminals[i]
                        ## is 1 if the state is the last in its trajectory, and
                        ## 0 otherwise.
                    ## HINT 2: self.gae_lambda is the lambda value in the
                        ## GAE formula
                    mask = 1 - terminals[i]
                    delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
                    advantages[i] = delta + self.gamma * self.gae_lambda * mask * advantages[i + 1]

                # remove dummy advantage
                advantages = advantages[:-1]

            else:
                ## TODO: compute advantage estimates using q_values, and values as baselines
                advantages = q_values_array - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            ## TODO: standardize the advantages to have a mean of zero
            ## and a standard deviation of one
            if isinstance(advantages, list):
                advantages = np.concatenate(advantages)
            else:
                advantages = np.asarray(advantages)

            adv_mean = np.mean(advantages)
            advantages = advantages - adv_mean
            adv_std = np.std(advantages)
            if adv_std > 1e-8:
                advantages = advantages / adv_std

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: array where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        rewards = np.asarray(rewards)
        if rewards.size == 0:
            return rewards

        discounts = np.power(self.gamma, np.arange(rewards.size))
        total_return = np.dot(discounts, rewards)
        discounted_returns = np.full_like(rewards, total_return)

        return discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns an array where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        rewards = np.asarray(rewards)
        if rewards.size == 0:
            return rewards
        
        discounted_cumsums = np.empty_like(rewards)
        running_return = 0.0
        gamma = self.gamma
        
        for idx in range(rewards.size - 1, -1, -1):
            running_return = rewards[idx] + gamma * running_return
            discounted_cumsums[idx] = running_return

        return discounted_cumsums
