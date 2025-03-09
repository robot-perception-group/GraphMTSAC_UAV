import datetime
import re
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

import wandb
from common.util import AverageMeter, check_act, check_obs, dump_cfg, np2ts
from common.vec_buffer import (FrameStackedReplayBuffer,
                               VecPrioritizedReplayBuffer,
                               VectorizedReplayBuffer)
from env.wrapper.helper import env_constructor

warnings.simplefilter("once", UserWarning)
exp_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


class AbstractAgent(ABC):
    @abstractmethod
    def act(self, s):
        pass

    @abstractmethod
    def step(self):
        pass


class IsaacAgent(AbstractAgent):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.env_cfg = cfg["env"]
        self.agent_cfg = cfg["agent"]
        self.buffer_cfg = cfg["buffer"]
        self.device = cfg["rl_device"]

        self.env, self.feature = env_constructor(
            env_cfg=self.env_cfg, device=self.device
        )
        assert (
            self.feature.dim == self.env.task.dim
        ), "feature and task dimension mismatch"

        self.n_env = self.env_cfg["num_envs"]
        self.max_episode_length = self.env_cfg["max_episode_length"]
        self.total_episodes = int(self.env_cfg["total_episodes"])
        self.total_timesteps = (
            self.n_env * self.max_episode_length * self.total_episodes
        )

        self.log_interval = self.agent_cfg["log_interval"]
        self.eval = self.agent_cfg["eval"]
        self.eval_interval = self.agent_cfg["eval_interval"]
        self.eval_episodes = self.agent_cfg["eval_episodes"]
        self.save_model = self.agent_cfg["save_model"]
        self.save_all = self.agent_cfg.get("save_all", True)
        self.save_best = self.agent_cfg.get("save_best", True)

        self.observation_dim = self.env.num_obs
        self.feature_dim = self.feature.dim
        self.action_dim = self.env.num_act
        self.observation_shape = [self.observation_dim]
        self.feature_shape = [self.feature_dim]
        self.action_shape = [self.action_dim]

        if self.buffer_cfg["prioritized_replay"]:
            self.replay_buffer = VecPrioritizedReplayBuffer(
                device=self.device,
                **self.buffer_cfg,
            )
        else:
            self.replay_buffer = VectorizedReplayBuffer(
                self.observation_shape,
                self.action_shape,
                device=self.device,
                **self.buffer_cfg,
            )
        self.mini_batch_size = int(self.buffer_cfg["mini_batch_size"])
        self.min_n_experience = int(self.buffer_cfg["min_n_experience"])

        self.gamma = int(self.agent_cfg["gamma"])
        self.updates_per_step = int(self.agent_cfg["updates_per_step"])
        self.reward_scale = int(self.agent_cfg["reward_scale"])

        if self.save_model:
            log_dir = (
                self.agent_cfg["name"]
                + "/"
                + self.env_cfg["env_name"]
                + "/"
                + exp_date
                + "/"
            )
            self.log_path = self.agent_cfg["log_path"] + log_dir
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
            dcfg = DictConfig(cfg)
            dcfg = OmegaConf.to_object(dcfg)
            dump_cfg(self.log_path + "cfg", dcfg)

        self.steps = 0
        self.episodes = 0
        self.eval_best_return = -float('inf')

        self.games_to_track = 100
        self.game_rewards = AverageMeter(1, self.games_to_track).to(self.device)
        self.game_lengths = AverageMeter(1, self.games_to_track).to(self.device)
        self.avgStepRew = AverageMeter(1, 20).to(self.device)

    def run(self):
        while True:
            training_result = self.train_episode()
            self._log_training(*training_result)

            if self.eval and (self.episodes % self.eval_interval == 0):
                evaluation_result = self.evaluate()
                returns = evaluation_result[0]
                mean_eval = torch.mean(returns).item()

                # update best evaluation return
                if mean_eval>self.eval_best_return:
                    better = True
                    self.eval_best_return = mean_eval
                else:
                    better = False

                self._log_evaluation(*evaluation_result)

                if self.save_model:
                    self._save_model(better)

            if self.episodes >= self.total_episodes:
                break

    def _log_training(self, episode_r, episode_steps, info):
        wandb.log({"reward/train": self.game_rewards.get_mean()})
        wandb.log({"reward/episode_length": self.game_lengths.get_mean()})

    def _log_evaluation(self, returns, episode_r):
        wandb.log({"reward/eval": torch.mean(returns).item()})
        wandb.log({"reward/eval_best": self.eval_best_return})

    def _save_model(self, better=False):
        if self.save_all:
            self.save_torch_model(f"model{self.episodes}/")
        if self.save_best and better:
            self.save_torch_model("best/")

    def train_episode(self, gui_app=None, gui_rew=None):
        self.episodes += 1
        episode_r = episode_steps = 0
        done = False

        print("episode = ", self.episodes)
        self.env.set_task("train")
        self.env.task.rand_task()

        s = self.reset_env()
        for _ in range(self.max_episode_length):
            episodeLen = self.env.progress_buf.clone()

            s_next, r, done = self.step(episode_steps, s)

            s = s_next
            self.steps += self.n_env
            episode_steps += 1
            episode_r += r

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.size()[0]:
                self.game_rewards.update(episode_r[done_ids])
                self.game_lengths.update(episodeLen[done_ids])

            # call gui update loop
            if gui_app:
                gui_app.update_idletasks()
                gui_app.update()
                self.avgStepRew.update(r)
                gui_rew.set(self.avgStepRew.get_mean())

            if episode_steps >= self.max_episode_length:
                break

        return episode_r, episode_steps, {}

    def step(self, episode_steps, s):
        assert not torch.isnan(
            s
        ).any(), f"detect anomaly state {(torch.isnan(s)==True).nonzero()}"

        assert not torch.isnan(
            self.env.task.Train.W
        ).any(), f"detect anomaly tasks {(torch.isnan(self.env.task.Train.W)==True).nonzero()}"

        a = self.act(s, self.env.task.Train)

        assert not torch.isnan(
            a
        ).any(), f"detect anomaly action {(torch.isnan(a)==True).nonzero()}"

        self.env.step(a)
        done = self.env.reset_buf.clone()
        s_next = self.env.obs_buf.clone()
        self.env.reset()

        assert not torch.isnan(
            s_next
        ).any(), f"detect anomaly state {(torch.isnan(s_next)==True).nonzero()}"

        r = self.calc_reward(s_next, a, self.env.task.Train.W)

        masked_done = False if episode_steps >= self.max_episode_length else done
        self.save_to_buffer(s, a, r, s_next, done, masked_done)

        if self.is_update():
            for _ in range(self.updates_per_step):
                self.learn()

        return s_next, r, done

    def is_update(self):
        return (
            len(self.replay_buffer) > self.mini_batch_size
            and self.steps >= self.min_n_experience
        )

    def reset_env(self):
        s = self.env.obs_buf.clone()
        if s is None:
            s = torch.zeros((self.n_env, self.env.num_obs))

        return s

    def save_to_buffer(self, s, a, r, s_next, done, masked_done):
        r = r[:, None] * self.reward_scale
        done = done[:, None]
        masked_done = masked_done[:, None]

        self.replay_buffer.add(s, a, r, s_next, masked_done)

    def evaluate(self):
        episodes = int(self.eval_episodes)
        if episodes == 0:
            return

        self.env.set_task("eval")

        print(
            f"===== evaluate at episode: {self.episodes} for {self.max_episode_length} steps ===="
        )

        returns = torch.zeros((episodes,), dtype=torch.float32)
        for i in range(episodes):
            episode_r = 0.0

            s = self.reset_env()
            for _ in range(self.max_episode_length):
                a = self.act(s, self.env.task.Eval, "exploit")
                self.env.step(a)
                s_next = self.env.obs_buf.clone()
                self.env.reset()

                r = self.calc_reward(s_next, a, self.env.task.Eval.W)

                s = s_next
                episode_r += r

            returns[i] = torch.mean(episode_r).item()

        print(f"evaluation return: ", returns)
        print(f"===== finish evaluate ====")

        return returns, episode_r
    
    def act(self, s, task, mode="explore"):
        s = check_obs(s, self.observation_dim)

        a = self._act(s, task, mode)

        a = check_act(a, self.action_dim)
        return a

    def _act(self, s, task, mode):
        with torch.no_grad():
            if (self.steps <= self.min_n_experience) and mode == "explore":
                a = self.random_action()

            if mode == "explore":
                a = self.explore(s, task.W)
            elif mode == "exploit":
                a = self.exploit(s, task.W)
        return a

    def random_action(self):
        return 2 * torch.rand((self.n_env, self.env.num_act), device=self.device) - 1

    def calc_reward(self, s, a, w):
        f = self.feature.extract(s, a)
        r = self._calc_reward(w, f)
        return r

    @torch.jit.script
    def _calc_reward(w, f):
        return torch.sum(w * f, 1)

    def explore(self):
        raise NotImplementedError

    def exploit(self):
        raise NotImplementedError

    def learn(self):
        pass

    def save_torch_model(self):
        raise NotImplementedError

    def load_torch_model(self):
        raise NotImplementedError


class MultitaskAgent(IsaacAgent):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # ENV observation is a vector composed of [ S | G | E ]
        self.state_dim = self.env.num_state  
        self.goal_dim = self.env.num_goal if hasattr(self.env, "num_goal") else 0 # G
        self.goal_idx = self.state_dim  

        self.env_latent_dim = self.env.num_latent if hasattr(self.env, "num_latent") else 0 # E
        self.env_latent_idx = self.state_dim + self.goal_dim 

        # curriculum learning
        self.curriculum = cfg["agent"].get("curriculum", False)
        if self.curriculum:
            self.difficulty, self.end_difficulty = cfg["agent"].get("difficulty_range", [0, 1])
            self.env.set_difficulty(self.difficulty)

            self.difficulty_stepsize = (self.end_difficulty - self.difficulty) / self.total_episodes

        self.log_task_returns = cfg["agent"].get("log_task_returns", True)

    def parse_observation(
        self,
        o: Tensor,
        combine_state_and_goal: bool = True
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Parse the observation tensor into state, goal, environment latent tensors.

        Args:
            o (Tensor): The observation tensor of shape [B, N], where B is the batch size
                and N is the dimensionality of the observation.
            combine_state_and_goal (bool): If True, the state and goal parts of the observation 
                are combined into a single tensor `s`. If False, `s` and `g` are returned separately.

        Returns:
            Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
                A tuple (s, g, e) where:
                - s (Tensor): The state part of the observation. If `combine_state_and_goal` is True, 
                this will include both state and goal.
                - g (Optional[Tensor]): The goal part of the observation if `combine_state_and_goal` is False, otherwise None.
                - e (Optional[Tensor]): The environment latent part of the observation if present, otherwise None.
        """
        if o.shape[1] > self.env_latent_idx:
            # parse observation if env_latent is included
            e = o[:, self.env_latent_idx :]
        else:
            # env_latent are not included
            e = None

        if combine_state_and_goal:
            s = o[:, : self.env_latent_idx]
            g = None
        else:
            s, g = (
                o[:, : self.goal_idx],
                o[:, self.goal_idx : self.env_latent_idx],
            )

        return s, g, e

    def _act(self, o, task, mode, combine_goal_and_state=True):
        with torch.no_grad():
            if (self.steps <= self.min_n_experience) and mode == "explore":
                a = self.random_action()

            s, g, e = self.parse_observation(o, combine_goal_and_state)
            if mode == "explore":
                a = self.explore(s, g, e, task.W, task.id)
            elif mode == "exploit":
                a = self.exploit(s, g, e, task.W, task.id)
        return a
    
    def train_episode(self, gui_app=None, gui_rew=None):
        # curriculum learning
        if (
            self.curriculum
            and self.difficulty < self.end_difficulty
        ):
            self.difficulty += self.difficulty_stepsize
            self.difficulty = min(self.difficulty, self.end_difficulty)
            self.env.set_difficulty(self.difficulty)

        episode_r, episode_steps, info = super().train_episode(gui_app, gui_rew)

        return episode_r, episode_steps, info

    def evaluate(self):
        returns, episode_r = super().evaluate()
        return returns, episode_r

    def _log_training(self, episode_r, episode_steps, info):
        # log overall training return
        super()._log_training(episode_r, episode_steps, info)

        # log difficulty during curriculum learning
        if self.curriculum:
            wandb.log(
                {
                    "env_state/difficulty": self.difficulty,
                }
            )


        # log return for each task
        if self.log_task_returns:
            task_return = self.env.task.get_taskR(episode_r)
            task_return = task_return.detach().tolist()
            for i in range(len(task_return)):
                wandb.log(
                    {
                        f"reward/task_return{i}": task_return[i],
                    }
                )

    def _log_evaluation(self, returns, episode_r):
        # log overall evaluation return
        super()._log_evaluation(returns, episode_r)

        # log return for each task
        if self.log_task_returns:
            task_return = self.env.task.get_taskR(episode_r, "eval")
            task_return = task_return.detach().tolist()

            for i in range(len(task_return)):
                wandb.log(
                    {
                        f"reward/eval_task_return{i}": task_return[i],
                    }
                )
