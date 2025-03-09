import torch
import re
from common.torch_jit_utils import *


class TaskObject:
    def __init__(self, n_env, taskSet, device) -> None:
        self.n_env = n_env
        self.device = device

        self.taskSet = self.normalize_task(
            torch.tensor(taskSet, dtype=torch.float32, device=self.device)
        )
        self.dim = int(self.taskSet.shape[1])

        # initialize task ratio to uniform distribution and then sample task for each environment
        self.reset_taskRatio()
        self.sample_tasks()

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, w: torch.tensor):
        """manually set task"""
        self._W = self.normalize_task(w)

    @property
    def id(self):
        return self._id

    def sample_tasks(self):
        """sample tasks based on the task ratio and normalize the task weight"""
        self._id = self.sample_taskID(self.taskRatio)
        self._W = self.taskSet[self.id]

    def sample_taskID(self, ratio):
        """sample tasks id based on the task ratio"""
        return torch.multinomial(ratio, self.n_env, replacement=True)

    def normalize_task(self, w):
        w /= w.norm(1, 1, keepdim=True) + 1e-6
        return w

    def reset_taskRatio(self):
        self.taskRatio = torch.ones(len(self.taskSet), device=self.device) / len(
            self.taskSet
        )

    def add_task(self, w: torch.tensor):
        w = w.view(-1, self.dim)
        w = self.normalize_task(w)
        self.taskSet = torch.cat([self.taskSet, w], 0)
        self.reset_taskRatio()
        self.sample_tasks()

class SmartTask:
    def __init__(self, env_cfg, device) -> None:
        env_cfg = env_cfg
        task_cfg = env_cfg["task"]

        self.feature_cfg = env_cfg["feature"]
        self.device = device
        self.n_env = env_cfg["num_envs"]
        self.verbose = task_cfg.get("verbose", False)
        self.adaptive_task = task_cfg.get("adaptive_task", False)

        trainSet = task_cfg["taskSet"]["train"]
        trainSet = re.split(r"\s*,\s*", trainSet)
        trainTaskSet = []
        self.n_trainTaskSet = 0
        for s in trainSet:
            trainTaskSet += task_cfg["taskSet"][s]
            self.n_trainTaskSet += 1
        self.Train = TaskObject(self.n_env, trainTaskSet, device)

        evalSet = task_cfg["taskSet"]["eval"]
        evalSet = re.split(r"\s*,\s*", evalSet)
        evalTaskSet = []
        self.n_evalTaskSet = 0
        for s in evalSet:
            evalTaskSet += task_cfg["taskSet"][s]
            self.n_evalTaskSet += 1
        self.Eval = TaskObject(self.n_env, evalTaskSet, device)

        self.dim = int(self.Train.dim)

        if self.verbose:
            print("[Task] training tasks id: \n", self.Train.id)
            print("[Task] training tasks: \n", self.Train.W)
            print("[Task] evaluation tasks id: \n", self.Eval.id)
            print("[Task] evaluation tasks: \n", self.Eval.W)
            print("\n")

    def rand_task(self):
        self.Train.sample_tasks()

        if self.verbose:
            print("[Task] sample new tasks:")
            print("[Task] Train.W: ", self.Train.W)
            print("[Task] Train.taskRatio: ", self.Train.taskRatio)
            print("[Task] Train TaskID Counts: ", torch.bincount(self.Train.id))
            print("[Task] Eval.W: ", self.Eval.W)
            print("[Task] Eval.taskRatio: ", self.Eval.taskRatio)
            print("[Task] Eval TaskID Counts: ", torch.bincount(self.Eval.id))
            print("\n")

    def get_taskR(self, episode_r, mode="train"):
        """
        return the episode return of each task based on the task ratio of the taskObj
        """
        if mode == "eval":
            taskR = self._get_taskR(self.Eval, episode_r)
        else:
            taskR = self._get_taskR(self.Train, episode_r)

        if self.adaptive_task:
            self._adapt_task(taskR)

        return taskR

    def _get_taskR(self, taskObj, episode_r):
        """compute individual task return based on the current task id"""
        tasksR = torch.zeros_like(taskObj.taskRatio, device=self.device)
        tasksCnt = torch.zeros_like(taskObj.taskRatio, device=self.device)
        id = taskObj.id

        tasksR = tasksR.index_add(dim=0, index=id, source=episode_r.float())

        cnt = torch.bincount(id)
        tasksCnt[: cnt.shape[0]] += cnt

        return tasksR / (tasksCnt + 1e-6)

    def _adapt_task(self, TaskReturn):
        """
        Update task ratio based on reward.
        The more reward the less likely for a task to be sampled.
        """
        new_ratio = (TaskReturn + 1e-6) ** -1
        new_ratio /= new_ratio.norm(1, keepdim=True)
        self.Train.taskRatio = new_ratio

        if self.verbose:
            print(
                f"[Task] updated task ratio: {new_ratio} \n as inverse of return {TaskReturn} \n"
            )

    def add_trainTasks(self, w: torch.tensor):
        self.Train.add_task(w)
        if self.verbose:
            print(f"[Task] new tasks {w} added to train task set \n")

    def add_evalTasks(self, w: torch.tensor):
        self.Eval.add_task(w)
        if self.verbose:
            print(f"[Task] new tasks {w} added to evaluation task set \n")