import torch
from torch.optim import Adam

import wandb
from common.util import *
from common.vec_buffer import FIFOBuffer
from common.model.feature_extractor import AdaptationModule, FCN

from .mtsac import MultitaskSACAgent

class RMAMultitaskSACAgent(MultitaskSACAgent):
    def __init__(self, cfg):

        self.agent_cfg = cfg["agent"]

        self.rma = self.agent_cfg["RMA"]
        self.rma_phase = self.agent_cfg.get("phase", 1)

        self.latent_dim = self.agent_cfg["latent_dim"]
        if not self.latent_dim:
            self.latent_dim = int(self.env_latent_dim // 2)  # Z = E/2

        # RMA only works with domain randomization
        cfg["env"]["dynamics"]["domain_rand"] = True

        super().__init__(cfg)
        
        self.episodes_phase2 = int(self.total_episodes // 2)
        self.eval_best_return_phase2 = -float('inf')

        self.prev_traj = FIFOBuffer(
            n_env=self.n_env,
            traj_dim=self.state_dim, # S+G+Z
            stack_size=self.stack_size - 1,
            device=self.device,
        )

    def _instantiate_model(self):
        observation_dim = self.state_dim + self.goal_dim + self.latent_dim
        super()._instantiate_critic(observation_dim)

        if self.policy_net_kwargs["architecture"]=="graph":
            latent_start = self.state_dim + self.goal_dim
            self.num_modalities = 0
            for _, modal_type in self.env.observation_modal_slice_and_type:
                self.num_modalities = max(self.num_modalities, modal_type)
            latent_modals = (slice(latent_start, latent_start+self.latent_dim), self.num_modalities+1)

            modal = self.env.observation_modal_slice_and_type+[(latent_modals)]
        else:
            modal = None
        super()._instantiate_policy(observation_dim, modal)

        self.encoder_net_kwargs = self.agent_cfg["encoder_net_kwargs"]
        self.encoder = FCN(
            in_dim=self.env_latent_dim,
            out_dim=self.latent_dim,
            **self.encoder_net_kwargs,
        ).to(self.device)


    def _instantiate_adaptor(self):
        self.adaptor_net_kwargs = self.agent_cfg["adaptor_net_kwargs"]
        self.adaptor_lr = self.agent_cfg["adaptor_lr"]
        self.adaptor = AdaptationModule(
            in_dim=self.state_dim,
            out_dim=self.latent_dim,
            stack_size=self.stack_size - 1,
            **self.adaptor_net_kwargs,
        ).to(self.device)

        self.adaptor_optimizer = Adam(
            self.adaptor.parameters(), lr=self.adaptor_lr, weight_decay=self.weight_decay
        )

    def _instantiate_optimizer(self):
        params = [
            {"params": self.critic.parameters()},
            {"params": self.encoder.parameters()},
        ]
        self.q_optimizer = Adam(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=self.policy_lr, weight_decay=self.weight_decay
        )


    def run(self):
        if self.rma_phase == 1:  # train with encoder
            grad_true(self.critic)
            grad_false(self.critic_target)
            grad_true(self.policy)
            grad_true(self.encoder)

            self.train_phase(self.episodes + self.total_episodes)
            self.rma_phase += 1

        if self.rma_phase == 2 and self.rma:  # train with adaptor
            self._instantiate_adaptor()

            if self.save_model and not self.load_model:
                # if models were saved previously
                # and no model is selected manually, load best model from phase1
                model_path = self.log_path + "phase1_best/"
                self.load_torch_model(model_path)

            grad_false(self.critic)
            grad_false(self.critic_target)
            grad_false(self.policy)
            grad_false(self.encoder)
            grad_true(self.adaptor)

            self.train_phase(self.episodes + self.episodes_phase2)
            self.rma_phase += 1

    def train_phase(self, episodes):
        print(f"============= start phase {self.rma_phase} =============")
        while True:
            training_result = self.train_episode()
            self._log_training(*training_result)

            if self.eval and (self.episodes % self.eval_interval == 0):
                evaluation_result = self.evaluate()
                returns = evaluation_result[0]
                mean_eval = torch.mean(returns).item()

                # update best evaluation return
                if self.rma_phase == 1:
                    better = mean_eval > self.eval_best_return
                    if better:
                        self.eval_best_return = mean_eval
                elif self.rma_phase == 2:
                    better = mean_eval > self.eval_best_return_phase2
                    if better:
                        self.eval_best_return_phase2 = mean_eval
                
                self._log_evaluation(*evaluation_result)

                if self.save_model:
                    self._save_model(better)

            if self.episodes >= episodes:
                break

    def _act(self, o, task, mode):
        a = super()._act(o, task, mode)

        if self.rma_phase != 1:
            s, _, _ = self.parse_observation(o, combine_state_and_goal=False) # [N, S]
            done = self.env.reset_buf.clone()
            self.prev_traj.add(s, done)

        return a


    def explore(self, s, g, e, w, id):  # act with randomness
        with torch.no_grad():
            s = self.encode_state(s, e) # [N, S+G+Z]
            a, _, _ = self.policy.sample(s, w)
        return a

    def exploit(self, s, g, e, w, id):  # act without randomness
        with torch.no_grad():
            s = self.encode_state(s, e) # [N, S+G+Z]
            _, _, a = self.policy.sample(s, w)
        return a

    def encode_state(self, s_raw, e):
        if self.rma_phase == 1:
            z = self.encoder(e)  # [N, Z] <-- [N, E]
        else:
            z = self.adaptor(self.prev_traj.get())  # [N, Z] <-- [N, S, K-1]

        z[:,1] = -1 
        # z[:,0], z[:,1] = -0.083, -0.0716 # 234
        # z[:,0], z[:,1] = -0.0238, -0.0566 # 345
        # z[:,0], z[:,1] = -0.132, -0.0746 # 456
        s = torch.concat([s_raw, z], dim=1)  # [N, S+G+Z]
        return s

    def reset_env(self):
        self.prev_traj.clear()
        return super().reset_env()

    def learn(self):
        if self.rma_phase == 1:
            self.learn_phase1()
        elif self.rma_phase == 2:
            self.learn_phase2()
        else:
            pass

    def learn_phase1(self):
        super().learn()

    def learn_phase2(self):
        self.learn_steps += 1
        batch = self.replay_buffer.sample(self.mini_batch_size)
        adaptor_loss, info = self.update_adaptor(batch)

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/adaptor": adaptor_loss,
                "state/lr_adaptor": self.adaptor_optimizer.param_groups[0]["lr"],
            }

            wandb.log(metrics)

    def update_critic(self, batch):
        (s, a, s_next, dones) = (
            batch["obs"],
            batch["action"],
            batch["next_obs"],
            batch["done"],
        )
        s, _, e = self.parse_observation(s)  # [N, S+G], [N, E]
        s_next, _, e_next = self.parse_observation(s_next)  # [N, S+G], [N, E]

        # log values to monitor training.
        info = {}

        z = self.encoder(e)
        z_next = self.encoder(e_next)

        s = torch.concat([s, z], dim=1)  # [N, S+G+Z]
        s_next = torch.concat([s_next, z_next], dim=1)  # [N, S+G+Z]

        # pair w and (s,a,s') to same size, w: T-->N*T, (s,a,s'): N-->N*T
        s, a, s_next, dones, w = get_sasdw_pairs(s, a, s_next, dones, self.w_primitive)

        # compute target value and update critic network
        q_loss, info = self._update_critic(s, a, s_next, dones, w, info)

        return q_loss, info

    def update_policy(self, batch):
        s = batch["obs"]
        s, _, e  = self.parse_observation(s)  # [N, S], [N, E]

        z = self.encoder(e)  # [N, Z]
        s = torch.concat([s, z], dim=1)  # [N, S+Z]

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
            s_next, _, e_next  = self.parse_observation(s_next)  # [N, S], [N, E]
            z_next = self.encoder(e_next)  # [N, Z]
            s_next = torch.concat([s_next, z_next], dim=1)  # [N, S+Z]

            policy_loss, info = self._apply_action_smoothness_loss(s, info, w, sampled_a, policy_loss, s_next)

        policy_loss = torch.mean(policy_loss)
        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)

        return policy_loss.detach().item(), entropy, info

    def update_adaptor(self, batch):
        s, s_next = batch["obs"], batch["next_obs"]

        _, _, e = self.parse_observation(s)
        _, _, e_next = self.parse_observation(s_next)

        z = self.encoder(e)
        z_next = self.encoder(e_next)

        s_stack, _, _ = self.parse_observation(batch["stacked_obs"], False)  # [N,S,K] <-- [N, S+E, K]
        z_hat = self.adaptor(s_stack[:, :, 1:])  # [N, S, K-1]
        z_next_hat = self.adaptor(s_stack[:, :, :-1])  # [N, S, K-1]

        adaptor_loss = self.mse_loss(z_hat, z) + self.mse_loss(z_next_hat, z_next)

        self.adaptor_optimizer.zero_grad()
        adaptor_loss.backward()
        self.adaptor_optimizer.step()

        info = {}
        info["adaptor_loss"] = adaptor_loss.detach().item()

        return adaptor_loss.detach().item(), info

    def _save_model(self, better=False):
        if self.save_all:
            self.save_torch_model(f"phase{self.rma_phase}_model{self.episodes}/")
        if self.save_best and better:
            if self.rma_phase == 1:
                self.save_torch_model(f"phase1_best/")
            elif self.rma_phase == 2:
                self.save_torch_model(f"phase2_best/")

    def save_torch_model(self, folder_name):
        from pathlib import Path

        path = self.log_path + folder_name
        Path(path).mkdir(parents=True, exist_ok=True)

        self.policy.save(path + "policy")
        self.critic.save(path + "critic")
        self.encoder.save(path + "encoder")

        # adaptor is instantiated after phase 2
        if self.rma_phase >= 2:
            self.adaptor.save(path + "adaptor")

        print("model saves at: ", path)

    def load_torch_model(self, path):
        self.policy.load(path + "policy")
        self.critic.load(path + "critic")
        self.encoder.load(path + "encoder")

        # adaptor is instantiated after phase 2
        # it is only loaded in evaluation phase (phase 3), 
        if self.rma_phase == 3: 
            self._instantiate_adaptor()
            self.adaptor.load(path + "adaptor")

        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)

        print("model loaded from: ", path)

    def _log_training(self, episode_r, episode_steps, info):
        # log difficulty during curriculum learning
        if self.curriculum:
            wandb.log(
                {
                    "env_state/difficulty": self.difficulty,
                }
            )


        # log overall training return
        wandb.log(
            {
                f"phase{self.rma_phase}_reward/train": self.game_rewards.get_mean(),
                f"phase{self.rma_phase}_reward/episode_length": self.game_lengths.get_mean(),
            }
        )

        # log return for each task
        if self.log_task_returns:
            task_return = self.env.task.get_taskR(episode_r)
            task_return = task_return.detach().tolist()
            for i in range(len(task_return)):
                wandb.log(
                    {
                        f"phase{self.rma_phase}_reward/task_return{i}": task_return[i],
                    }
                )

    def _log_evaluation(self, returns, episode_r):
        # log overall evaluation return
        wandb.log({f"phase{self.rma_phase}_reward/eval": torch.mean(returns).item()})

        if self.rma_phase==1:
            wandb.log({f"phase1_reward/eval_best": self.eval_best_return})
        elif self.rma_phase==2:
            wandb.log({f"phase2_reward/eval_best": self.eval_best_return_phase2})

        # log return for each task
        if self.log_task_returns:
            task_return = self.env.task.get_taskR(episode_r, "eval")
            task_return = task_return.detach().tolist()

            for i in range(len(task_return)):
                wandb.log(
                    {
                        f"phase{self.rma_phase}_reward/eval_task_return{i}": task_return[i],
                    }
                )
