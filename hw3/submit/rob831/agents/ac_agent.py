from collections import OrderedDict

from rob831.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from rob831.infrastructure.replay_buffer import ReplayBuffer
from rob831.infrastructure.utils import *
from rob831.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # advantage = estimate_advantage(...)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        critic_losses = []
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            critic_losses.append(critic_loss)
            
        adv_n = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        
        actor_losses = []
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss = self.actor.update(ob_no, ac_na, adv_n=adv_n)
            actor_losses.append(actor_loss)
        
        loss = OrderedDict()
        loss['Loss_Critic'] = np.mean(critic_losses) if critic_losses else 0.0
        loss['Loss_Actor'] = np.mean(actor_losses) if actor_losses else 0.0

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        
        
        v_s = np.asarray(self.critic.forward_np(ob_no))
        v_next = np.asarray(self.critic.forward_np(next_ob_no))
        rewards = np.asarray(re_n)
        terminals = np.asarray(terminal_n)
        
        non_terminal = 1 - terminals
        q_sa = rewards + self.gamma * v_next * non_terminal
        adv_n = q_sa - v_s

        if self.standardize_advantages:
            adv_mean = np.mean(adv_n)
            adv_std = np.std(adv_n)
            adv_n = adv_n - adv_mean
            if adv_std > 1e-8:
                adv_n = adv_n / (adv_std + 1e-8)
                
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
