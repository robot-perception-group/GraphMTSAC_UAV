import torch
import torch.nn as nn
from torch.optim import Adam

import wandb
from common.agent import MultitaskAgent
from common.model.policy import CompositionalGaussianPolicyNetwork, ParallelContextualGaussianPolicyNetwork, ParallelMTGaussianPolicyNetwork, MTGraphGaussianPolicyNetwork
from common.util import *
from common.model.value import CompositionalQNetwork, ParallelContextualQNetwork, ParallelMTQNetwork
from common.vec_buffer import FrameStackedReplayBuffer


class MultitaskSACAgent(MultitaskAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.wandb_verbose = self.agent_cfg.get("wandb_verbose", False)

        self.load_model = self.agent_cfg.get("load_model", False)
        self.model_path = self.agent_cfg.get("model_path", None)

        self.lr = self.agent_cfg["lr"]
        self.policy_lr = self.agent_cfg["policy_lr"]
        self.value_net_kwargs = self.agent_cfg["value_net_kwargs"]
        self.policy_net_kwargs = self.agent_cfg["policy_net_kwargs"]
        self.gamma = torch.tensor(self.agent_cfg["gamma"], device=self.device)
        self.tau = self.agent_cfg["tau"]
        self.td_target_update_interval = int(
            self.agent_cfg["td_target_update_interval"]
        )
        self.updates_per_step = self.agent_cfg["updates_per_step"]
        self.grad_clip = self.agent_cfg["grad_clip"]
        self.entropy_tuning = self.agent_cfg["entropy_tuning"]

        self.use_action_smoothness_loss=self.agent_cfg["use_action_smoothness_loss"]
        if self.use_action_smoothness_loss:
            self.asp = self.agent_cfg["action_smooth_params"]

        self.weight_decay=self.agent_cfg.get("weight_decay", 0.0)

        self.framestacked_replay = self.buffer_cfg["framestacked_replay"]
        self.stack_size = self.buffer_cfg["stack_size"]
        assert (
            self.framestacked_replay == True
        ), "This agent only support framestacked replay"

        # define primitive tasks
        self.w_primitive = self.env.task.Train.taskSet
        self.n_tasks = self.w_primitive.shape[0]
        self.n_taskSets = self.env.task.n_trainTaskSet

        self.replay_buffer = FrameStackedReplayBuffer(
            obs_shape=self.observation_shape,
            action_shape=self.action_shape,
            feature_shape=self.feature_shape,
            device=self.device,
            **self.buffer_cfg,
        )

        self._instantiate_model()

        if self.load_model and self.model_path is not None:
            self.model_path = self.agent_cfg["log_path"] + self.model_path
            print("load model:", self.model_path)
            self.load_torch_model(self.model_path)


        self._instantiate_optimizer()

        self.mse_loss = nn.MSELoss()

        if self.entropy_tuning:
            self.alpha_lr = self.agent_cfg["alpha_lr"]
            target_entropy_ratio = self.agent_cfg["target_entropy_ratio"]
            self.target_entropy = -target_entropy_ratio*torch.prod(
                torch.Tensor(self.action_shape).to(self.device)
            )  # target entropy = -|A|
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device
            )  # optimize log(alpha), instead of alpha
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(self.agent_cfg["alpha"]).to(self.device)

        self.learn_steps = 0

    def _instantiate_model(self):
        observation_dim = self.state_dim + self.goal_dim
        self._instantiate_critic(observation_dim)
        self._instantiate_policy(observation_dim, self.env.observation_modal_slice_and_type)

    def _instantiate_critic(self, observation_dim):
        if self.value_net_kwargs["architecture"]=="plain":
            self.critic, self.critic_target = [ParallelMTQNetwork(
                    observation_dim=observation_dim,
                    action_dim=self.action_dim,
                    task_dim=self.feature_dim,
                    **self.value_net_kwargs,
                ).to(self.device) for _ in range(2)]

        elif self.value_net_kwargs["architecture"]=="compositional":
            args = self.value_net_kwargs["composition"]
            self.critic, self.critic_target = [CompositionalQNetwork(
                    observation_dim=observation_dim,
                    action_dim=self.action_dim,
                    task_dim=self.feature_dim,
                    context_dim=args["context_dim"],
                    use_comp_layer=args["use_comp_layer"],
                    state_independent_context=args["state_independent_context"],
                    **self.value_net_kwargs,
                ).to(self.device) for _ in range(2)]

        elif self.value_net_kwargs["architecture"]=="parallel":
            args = self.value_net_kwargs["parallel"]
            self.critic, self.critic_target = [ParallelContextualQNetwork(
                    observation_dim=observation_dim,
                    action_dim=self.action_dim,
                    task_dim=self.feature_dim,
                    context_dim=args["context_dim"],
                    state_independent_context=args["state_independent_context"],
                    **self.value_net_kwargs,
                ).to(self.device) for _ in range(2)]

        else:
            raise NotImplementedError("choose value architecture from [plain, compositional, parallel]")

        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)

    def _instantiate_policy(self, observation_dim, obs_modal_slice=None):
        if self.policy_net_kwargs["architecture"]=="plain":
            self.policy = ParallelMTGaussianPolicyNetwork(
                    observation_dim=observation_dim,
                    action_dim=self.action_dim,
                    task_dim=self.feature_dim,
                    **self.policy_net_kwargs,
                ).to(self.device)

        elif self.policy_net_kwargs["architecture"]=="compositional":
            args = self.policy_net_kwargs["composition"]
            self.policy = CompositionalGaussianPolicyNetwork(
                    observation_dim=observation_dim,
                    action_dim=self.action_dim,
                    task_dim=self.feature_dim,
                    context_dim=args["context_dim"],
                    use_comp_layer=args["use_comp_layer"],
                    state_independent_context=args["state_independent_context"],
                    **self.policy_net_kwargs,
                ).to(self.device)

        elif self.policy_net_kwargs["architecture"]=="parallel":
            args = self.policy_net_kwargs["parallel"]
            self.policy = ParallelContextualGaussianPolicyNetwork(
                    observation_dim=observation_dim,
                    action_dim=self.action_dim,
                    task_dim=self.feature_dim,
                    context_dim=args["context_dim"],
                    state_independent_context=args["state_independent_context"],
                    **self.policy_net_kwargs,
                ).to(self.device)

        elif self.policy_net_kwargs["architecture"]=="graph":
            args = self.policy_net_kwargs["graph"]

            positive_edge = self.env_cfg["graph_with_task"]["positive_edge"]
            negative_edge = self.env_cfg["graph_with_task"]["negative_edge"]

            num_gcn_layers = args["num_gcn_layers"]
            embedding_dim = args["embedding_dim"]
            pos_edge_w = args["init_positive_edge_weight"]
            neg_edge_w = args["init_negative_edge_weight"]

            # create initial adjacency matrix
            num_nodes = observation_dim + self.feature_dim 
            adjacency_table = torch.zeros(num_nodes, num_nodes)

            # fill adjacency matrix with initial weights
            for action_node, state_node_list in positive_edge.items():
                for state_node in state_node_list:
                    if state_node < num_nodes:
                        adjacency_table[int(action_node)][int(state_node)] = pos_edge_w
                    else:
                        print("[Warning] state node out of range, likely because RMA is not enabled: ", state_node)

            for action_node, state_node_list in negative_edge.items():
                for state_node in state_node_list:
                    if state_node < num_nodes:
                        adjacency_table[int(action_node)][int(state_node)] = neg_edge_w
                    else:
                        print("[Warning] state node out of range, likely because RMA is not enabled: ", state_node)

            self.policy = MTGraphGaussianPolicyNetwork(
                observation_dim=observation_dim,
                action_dim=self.action_dim,
                task_dim=self.feature_dim,
                obs_modal_slice=obs_modal_slice,
                state_action_adjacency_matrix=adjacency_table,
                num_gcn_layers=num_gcn_layers,
                embedding_dim=embedding_dim, 
                **self.policy_net_kwargs,
            ).to(self.device)

        else:
            raise NotImplementedError("choose policy architecture from [plain, compositional, parallel, graph]")

        total_params = sum(p.numel() for p in self.policy.parameters())
        print("total number of policy parameters: ", total_params, ". size: ", total_params*4/1000, "[kB]")

    def _instantiate_optimizer(self):
        self.q_optimizer = Adam(
            self.critic.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=self.policy_lr, weight_decay=self.weight_decay
        )

    def explore(self, s, g, e, w, id):  # act with randomness
        with torch.no_grad():
            a, _, _ = self.policy.sample(s, w)
        return a

    def exploit(self, s, g, e, w, id):  # act without randomness
        with torch.no_grad():
            _, _, a = self.policy.sample(s, w)
        return a

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        batch = self.replay_buffer.sample(self.mini_batch_size)

        q_loss, info_q = self.update_critic(batch)
        policy_loss, entropies, info_pi = self.update_policy(batch)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(
                entropies, self.target_entropy, self.log_alpha
            )
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/Q": q_loss,
                "loss/policy": policy_loss,
                "state/mean_Q": info_q["mean_Q"],
                "state/entropy": entropies.detach().mean().item(),
            }
            if self.entropy_tuning:
                metrics.update(
                    {
                        "loss/alpha": entropy_loss.detach().item(),
                        "state/alpha": self.alpha.mean().detach().item(),
                    }
                )
            if self.use_action_smoothness_loss:
                metrics.update(
                    {
                        "loss/temporal_smoothness": info_pi["temporal_smoothness"],
                        "loss/spatial_smoothness": info_pi["spatial_smoothness"],
                    }
                )


            wandb.log(metrics)

    def update_critic(self, batch):
        (s, a, s_next, dones) = (
            batch["obs"],
            batch["action"],
            batch["next_obs"],
            batch["done"],
        )

        s, _, _ = self.parse_observation(s, combine_state_and_goal=True)  # [N, S], 
        s_next, _, _ = self.parse_observation(s_next, combine_state_and_goal=True)  # [N, S]

        # pair w and (s,a,s') to same size, w: T-->N*T, (s,a,s'): N-->N*T
        s, a, s_next, dones, w = get_sasdw_pairs(s, a, s_next, dones, self.w_primitive)

        
        # log values to monitor training.
        info = {}

        # compute target value and update critic network
        q_loss, info = self._update_critic(s, a, s_next, dones, w, info)

        return q_loss, info

    def _update_critic(self, s, a, s_next, dones, w, info):
        # derive feature from state
        f = self.feature.extract(s, a)  # [NT,F]

        # compute reward from feature
        r = self._compute_reward(w, f)  # [NT,1] <-- [NT,F], [NT, F]

        curr_qs = self.critic(s, a, w)
        target_q = self.calc_target_q(r, s_next, dones, w)

        # Critic loss is mean squared TD errors.
        q_loss = 0
        for i in range(curr_qs.shape[1]):
            q_loss += self.mse_loss(curr_qs[:, i], target_q)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        q_loss = q_loss.detach().item()
        info["mean_Q"] = curr_qs.detach().mean().item()
        return q_loss, info

    def update_policy(self, batch):
        s = batch["obs"]
        s, _, _ = self.parse_observation(s)  # [N, S]

        # log values to monitor training.
        info = {}

        # pair states with all tasks
        s, w = get_sw_pairs(s, self.w_primitive)  # [N*T,S], [N*T,F] <-- [N,S], [T,F]

        # We re-sample actions to calculate expectations of Q.
        sampled_a, entropy, _ = self.policy.sample(s, w)  # [N*T,A]

        # expectations of Q with clipped double Q technique
        qs = self.critic(s, sampled_a, w)
        q = self._min_qs(qs)

        # Policy objective is maximization of (Q + alpha * entropy).
        policy_loss = -q - self.alpha * entropy  # [N*T]

        if self.use_action_smoothness_loss:
            s_next = batch["next_obs"]
            s_next, _, _ = self.parse_observation(s_next)  # [N, S]

            policy_loss, info = self._apply_action_smoothness_loss(s, info, w, sampled_a, policy_loss, s_next)

        policy_loss = torch.mean(policy_loss)
        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)

        return policy_loss.detach().item(), entropy, info
    
    def calc_target_q(self, r, s_next, dones, w_next):
        with torch.no_grad():
            a_next, _, _ = self.policy.sample(s_next, w_next)
            next_qs = self.critic_target(s_next, a_next, w_next)
            next_q = self._min_qs(next_qs)

        assert(next_q.shape==r.shape, f"size mismatch between next_q {next_q} and r {r}")
        target_q = self._calc_target_q(r, dones, self.gamma, next_q)
        return target_q.squeeze(1)

    def _apply_action_smoothness_loss(self, s, info, w, sampled_a, policy_loss, s_next):
        s_next, w_next = get_sw_pairs(s_next, self.w_primitive) # [N*T,S]
        sampled_a_next, _, _ = self.policy.sample(s_next, w_next) # [N*T,A]
        temporal_smoothness = torch.norm(sampled_a-sampled_a_next, p=2, dim=-1, keepdim=True)

        noised_s = s + torch.normal(mean=0, std=0.001, size=s.shape, device=self.device)
        sampled_a_noised, _, _ = self.policy.sample(noised_s, w) # [N*T,A]
        spatial_smoothness = torch.norm(sampled_a-sampled_a_noised, p=2, dim=-1, keepdim=True)

        policy_loss += self.asp[0]*temporal_smoothness + self.asp[1]*spatial_smoothness

        info["temporal_smoothness"] = temporal_smoothness.mean().detach().item()
        info["spatial_smoothness"] = spatial_smoothness.mean().detach().item()

        return policy_loss, info
    
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


#####################################################################
###=========================jit functions=========================###
#####################################################################

    @torch.jit.script
    def calc_entropy_loss(entropy, target_entropy, log_alpha):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(log_alpha * (target_entropy - entropy).detach())
        return entropy_loss

    @torch.jit.script
    def _compute_reward(w, f):
        return torch.sum(f * w, 1, keepdim=True)

    @torch.jit.script
    def _min_qs(qs):
        return torch.min(qs, dim=1, keepdim=True)[0]

    @torch.jit.script
    def _calc_target_q(r, dones, gamma, next_q):
        return r + (~dones) * gamma * next_q