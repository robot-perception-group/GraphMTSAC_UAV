import torch
from torch.optim import Adam
import torch.nn as nn

import wandb
from common.agent import IsaacAgent
from common.model.policy import ParallelGaussianPolicyNetwork, GraphGaussianPolicyNetwork
from common.util import (
    grad_false,
    hard_update,
    soft_update,
    update_params,
)
from common.model.value import ParallelQNetwork


class SACAgent(IsaacAgent):
    """SAC
    Tuomas Haarnoja, Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.lr = self.agent_cfg["lr"]
        self.policy_lr = self.agent_cfg["policy_lr"]
        self.value_net_kwargs = self.agent_cfg["value_net_kwargs"]
        self.policy_net_kwargs = self.agent_cfg["policy_net_kwargs"]
        self.gamma = self.agent_cfg["gamma"]
        self.tau = self.agent_cfg["tau"]

        self.td_target_update_interval = int(
            self.agent_cfg["td_target_update_interval"]
        )
        self.grad_clip = self.agent_cfg["grad_clip"]
        self.entropy_tuning = self.agent_cfg["entropy_tuning"]

        self.critic, self.critic_target = [ParallelQNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            **self.value_net_kwargs,
        ).to(self.device) for _ in range(2)]

        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)

        self.mse_loss = nn.MSELoss()


        if self.policy_net_kwargs["architecture"]=="plain":
            self.policy = ParallelGaussianPolicyNetwork(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                **self.policy_net_kwargs,
            ).to(self.device)

        elif self.policy_net_kwargs["architecture"]=="graph":
            args = self.policy_net_kwargs["graph"]

            positive_edge = self.env_cfg["graph"]["positive_edge"]
            negative_edge = self.env_cfg["graph"]["negative_edge"]

            num_gcn_layers = args["num_gcn_layers"]
            embedding_dim = args["embedding_dim"]
            pos_edge_w = args["init_positive_edge_weight"]
            neg_edge_w = args["init_negative_edge_weight"]

            # create initial adjacency matrix
            num_nodes = self.observation_dim 
            adjacency_table = torch.zeros(num_nodes, num_nodes)

            # fill adjacency matrix with initial weights
            for action_node, state_node_list in positive_edge.items():
                for state_node in state_node_list:
                    if state_node < num_nodes:
                        adjacency_table[int(action_node)][int(state_node)] = pos_edge_w
                    else:
                        print("state node out of range: ", state_node)

            for action_node, state_node_list in negative_edge.items():
                for state_node in state_node_list:
                    if state_node < num_nodes:
                        adjacency_table[int(action_node)][int(state_node)] = neg_edge_w
                    else:
                        print("state node out of range: ", state_node)

            self.policy = GraphGaussianPolicyNetwork(
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    obs_modal_slice=self.env.observation_modal_slice_and_type,
                    state_action_adjacency_matrix=adjacency_table,
                    num_gcn_layers=num_gcn_layers,
                    embedding_dim=embedding_dim, 
                    **self.policy_net_kwargs,
                ).to(self.device)

        total_params = sum(p.numel() for p in self.policy.parameters())
        print("total number of policy parameters: ", total_params, ". size: ", total_params*4/1000, "[kB]")

        self.q_optimizer = Adam(
            self.critic.parameters(), lr=self.lr, betas=[0.9, 0.999]
        )
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=self.policy_lr, betas=[0.9, 0.999]
        )

        if self.entropy_tuning:
            self.alpha_lr = self.agent_cfg["alpha_lr"]
            self.target_entropy = -torch.prod(
                torch.Tensor(self.action_shape).to(self.device)
            ).item()  # target entropy = -|A|
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device
            )  # optimize log(alpha), instead of alpha
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(self.agent_cfg["alpha"]).to(self.device)

        self.learn_steps = 0

    def explore(self, s, w):  # act with randomness
        with torch.no_grad():
            a, _, _ = self.policy.sample(s)
        return a

    def exploit(self, s, w):  # act without randomness
        with torch.no_grad():
            _, _, a = self.policy.sample(s)
        return a

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        batch = self.replay_buffer.sample(self.mini_batch_size)

        q_loss, mean_q = self.update_critic(batch)
        policy_loss, entropies = self.update_policy(batch)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies)
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/Q": q_loss,
                "loss/policy": policy_loss,
                "state/mean_Q": mean_q,
                "state/entropy": entropies.detach().mean().item(),
            }
            if self.entropy_tuning:
                metrics.update(
                    {
                        "loss/alpha": entropy_loss.detach().item(),
                        "state/alpha": self.alpha.mean().detach().item(),
                    }
                )

            wandb.log(metrics)

    def update_critic(self, batch):
        (s, a, r, s_next, dones) = (
            batch["obs"],
            batch["action"],
            batch["reward"],
            batch["next_obs"],
            batch["done"],
        )

        curr_qs = self.critic(s, a)
        target_q = self.calc_target_q(r, s_next, dones)

        # Critic loss is mean squared TD errors.
        q_loss = 0
        for i in range(curr_qs.shape[1]):
            q_loss += self.mse_loss(curr_qs[:, i], target_q)

        update_params(self.q_optimizer, self.critic, q_loss, self.grad_clip)

        # log values to monitor training.
        q_loss = q_loss.detach().item()
        mean_qs = curr_qs.detach().mean().item()

        return q_loss, mean_qs

    def update_policy(self, batch):
        s = batch["obs"]

        # We re-sample actions to calculate expectations of Q.
        sampled_a, entropy, _ = self.policy.sample(s)
        # expectations of Q with clipped double Q technique
        qs = self.critic(s, sampled_a)

        # q = torch.min(qs, dim=1)[0]
        q = self._min_qs(qs)

        # Policy objective is maximization of (Q + alpha * entropy).
        policy_loss = torch.mean((-q - self.alpha * entropy))
        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)

        return policy_loss.detach().item(), entropy

    def calc_entropy_loss(self, entropy):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
        )
        return entropy_loss

    def calc_target_q(self, r, s_next, dones):
        with torch.no_grad():
            a_next, _, _ = self.policy.sample(s_next)
            next_qs = self.critic_target(s_next, a_next)
            next_q = self._min_qs(next_qs)

        target_q = r + (~dones) * self.gamma * next_q
        return target_q

    @torch.jit.script
    def _min_qs(qs):
        return torch.min(qs, dim=1, keepdim=False)[0]

    def save_torch_model(self, folder_name):
        from pathlib import Path

        path = self.log_path + folder_name
        Path(path).mkdir(parents=True, exist_ok=True)
        
        self.policy.save(path + "policy")
        self.critic.save(path + "critic")

        print("model saves at: ", path)

    def load_torch_model(self, path):
        self.policy.load(path + "policy")
        self.critic.load(path + "critic")

        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)

        print("model loaded from: ", path)
