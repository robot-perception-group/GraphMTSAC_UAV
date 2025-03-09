import torch
import re
import copy
from common.torch_jit_utils import *
from common.task.task import SmartTask

class TaskSetObj:
    """The task set object records the index of each task withint this task set
    and handles robot state sampling and reset conditions
    """

    def __init__(self, name, env_cfg, device) -> None:
        self.device = device
        self.TasksIdx = self._find_taskset_index(name, env_cfg)

    def reset_tasks(
        self,
        reset_buf,
        ids,
        **kwargs,
    ):
        """check whether to reset the state for this taskset"""
        condition = self._has_task_index(ids, repeat=False)
        return self._reset_tasks(reset_buf, condition, **kwargs)

    def _reset_tasks(self, reset_buf, condition, **kwargs):
        # Implement to set a different reset condition for the task set
        return reset_buf

    def sample_robot_state(self, ids, **kwargs):
        """sample robot state for this taskset"""
        condition = self._has_task_index(ids)
        return self._sample_state(condition, **kwargs)

    def _sample_state(
        self,
        condition,
        rb_pos,
        rb_vel,
        rb_ang,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        **kwargs,
    ):
        # Implement to sample different start condition for the task set
        return (
            rb_pos,
            rb_vel,
            rb_ang,
            rb_angvel,
            goal_pos,
            goal_vel,
            goal_velnorm,
            goal_ang,
            goal_angvel,
        )

    def _find_taskset_index(self, name, env_cfg):
        """find the taskset index by first counting non taskset index and then update every number in the list"""
        n_NonTasks = self._count_nontasks(name, env_cfg)
        n_Tasks = len(env_cfg["task"]["taskSet"][name])
        TasksIdx = [n_NonTasks + i for i in range(n_Tasks)]
        return TasksIdx

    def _count_nontasks(self, name, env_cfg):
        trainSet = env_cfg["task"]["taskSet"]["train"]
        trainSet = re.split(r"\s*,\s*", trainSet)

        n_NonTasks = 0
        for s in trainSet:
            if s == name:
                break
            n_NonTasks += len(env_cfg["task"]["taskSet"][s])
        return n_NonTasks

    def _has_task_index(self, ids, repeat=True):
        """takes env ids and find positions that is the same as task ids"""
        condition = sum(ids == i for i in self.TasksIdx).bool()

        if repeat:
            return condition[:, None].repeat(1, 3)
        else:
            return condition

class FWAttitudeTaskSetObj(TaskSetObj):
    """follow attitude target
    sample:
        - angle near attitude target
        - forward velocity
        - null angular velocity
    reset:
        - large angular velocity

    """

    def _sample_state(
        self,
        condition,
        rb_pos,
        rb_vel,
        rb_ang,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        **kwargs,
    ):
        zero_state = torch.zeros_like(rb_pos)

        # goal roll in 0deg, pitch in 90+-20deg, yaw in +-60deg
        att_goal_ang = torch.zeros_like(rb_ang)
        att_goal_ang[:, 1] = torch.pi / 2 + sampling(
            rb_ang.shape[0], torch.pi / 9, self.device
        )
        att_goal_ang[:, 2] = sampling(rb_ang.shape[0], torch.pi / 3, self.device)

        # goal_velnorm in range default goal_velnorm+-6m/s
        att_goal_velnorm = kwargs["target_velnorm"] + sampling(goal_velnorm.shape, 6.0, self.device)

        # robot roll in +-180deg, pitch in 90+-15deg, yaw in +-40
        att_ang = sampling(rb_ang.shape, torch.pi, self.device)
        att_ang[:, 1] = torch.pi / 2 + sampling(
            rb_ang.shape[0], torch.pi / 12, self.device
        )
        att_ang[:, 2] = sampling(rb_ang.shape[0], torch.pi / 4.5, self.device)

        # sample aoa and side slip angle in +-10deg
        ab_ang = att_ang + sampling(att_ang.shape, torch.pi / 18, self.device)
        rb_quat = quat_from_euler(ab_ang)
        forwardI = local_to_global(rb_quat, kwargs["forward"])

        # sample forward velocity in goal_vnorm+-6m/s
        vnorm = goal_velnorm + sampling((rb_ang.shape[0], 1), 6.0, self.device)
        att_vel = forwardI * vnorm

        rb_ang = torch.where(condition, att_ang, rb_ang)
        rb_vel = torch.where(condition, att_vel, rb_vel)
        rb_angvel = torch.where(condition, zero_state, rb_angvel)

        goal_ang = torch.where(condition, att_goal_ang, goal_ang)
        goal_velnorm = torch.where(
            condition[..., 0:1], att_goal_velnorm, goal_velnorm
        )
        goal_angvel = torch.where(condition, zero_state, goal_angvel)

        return (
            rb_pos,
            rb_vel,
            rb_ang,
            rb_angvel,
            goal_pos,
            goal_vel,
            goal_velnorm,
            goal_ang,
            goal_angvel,
        )

    def _reset_tasks(
        self,
        reset_buf,
        condition,
        rb_pos,
        rb_vel,
        rb_quats,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        **kwargs,
    ):
        ones = torch.ones_like(reset_buf)

        avel_norm = torch.norm(rb_angvel, p=2, dim=-1)

        reset_buf = torch.where(
            torch.logical_and(condition, (avel_norm > 10)), ones, reset_buf
        )
        return reset_buf


class TransitionTaskSetObj(TaskSetObj):
    """transition from qr pose to attitude tracking
    sample:
        - neutral state
    reset:
        - large angvel
    """

    def _sample_state(
        self,
        condition,
        rb_pos,
        rb_vel,
        rb_ang,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        **kwargs,
    ):
        zero_state = torch.zeros_like(rb_pos)

        # goal roll in 0deg, pitch in 90+-20deg, yaw in +-60deg
        att_goal_ang = torch.zeros_like(rb_ang)
        att_goal_ang[:, 1] = torch.pi / 2 + sampling(
            rb_ang.shape[0], torch.pi / 9, self.device
        )
        att_goal_ang[:, 2] = sampling(rb_ang.shape[0], torch.pi / 3, self.device)

        # rb angle in neutral state
        rb_ang[..., 0:2] = torch.where(
            condition[..., 0:2], zero_state[..., 0:2], rb_ang[..., 0:2]
        )
        rb_vel = torch.where(condition, zero_state, rb_vel)
        rb_angvel = torch.where(condition, zero_state, rb_angvel)

        goal_ang = torch.where(condition, att_goal_ang, goal_ang)
        goal_angvel = torch.where(condition, zero_state, goal_angvel)

        return (
            rb_pos,
            rb_vel,
            rb_ang,
            rb_angvel,
            goal_pos,
            goal_vel,
            goal_velnorm,
            goal_ang,
            goal_angvel,
        )

    def _reset_tasks(
        self,
        reset_buf,
        condition,
        rb_pos,
        rb_vel,
        rb_quats,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        **kwargs,
    ):
        ones = torch.ones_like(reset_buf)

        avel_norm = torch.norm(rb_angvel, p=2, dim=-1)

        reset_buf = torch.where(
            torch.logical_and(condition, (avel_norm > 10)), ones, reset_buf
        )
        return reset_buf


class PoseRecoverTaskSetObj(TaskSetObj):
    """recover from state to upright
    sample:
        - neutral goal
        - random angle
    """

    def _sample_state(
        self,
        condition,
        rb_pos,
        rb_vel,
        rb_ang,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        **kwargs,
    ):
        zero_state = torch.zeros_like(rb_pos)

        pr_ang = sampling(rb_ang.shape, torch.pi, self.device)

        rb_ang = torch.where(condition, pr_ang, rb_ang)
        goal_ang = torch.where(condition, zero_state, goal_ang)
        goal_vel = torch.where(condition, zero_state, goal_vel)
        goal_velnorm = torch.where(
            condition[..., 0].unsqueeze(1), torch.zeros_like(goal_velnorm), goal_velnorm
        )
        goal_angvel = torch.where(condition, zero_state, goal_angvel)

        return (
            rb_pos,
            rb_vel,
            rb_ang,
            rb_angvel,
            goal_pos,
            goal_vel,
            goal_velnorm,
            goal_ang,
            goal_angvel,
        )


class QRStabilizeTaskSetObj(TaskSetObj):
    """stabilization control in QR mode
    sample:
        - neutral state
    reset:
        - small vel, angvel
    """

    def _sample_state(
        self,
        condition,
        rb_pos,
        rb_vel,
        rb_ang,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        spawn_height,
        **kwargs,
    ):
        zero_state = torch.zeros_like(rb_pos)

        qr_pos = torch.zeros_like(rb_pos)
        qr_pos[..., 2] = spawn_height

        qr_ang = torch.zeros_like(rb_ang)
        qr_ang[..., 2] = rb_ang[..., 2]

        goal_qr_ang = torch.zeros_like(goal_ang)
        goal_qr_ang[..., 2] = rb_ang[..., 2]

        rb_pos = torch.where(condition, qr_pos, rb_pos)  # zero position
        rb_vel = torch.where(condition, zero_state, rb_vel) # zero velocity
        rb_ang = torch.where(condition, qr_ang, rb_ang) # zero roll pitch angle
        rb_angvel = torch.where(condition, zero_state, rb_angvel) # zero angvel

        goal_ang = torch.where(condition, goal_qr_ang, goal_ang) # zero roll pitch goal ang
        goal_vel = torch.where(condition, zero_state, goal_vel) # zero goal vel
        goal_velnorm = torch.where(
            condition[..., 0].unsqueeze(1), torch.zeros_like(goal_velnorm), goal_velnorm
        ) # zero goal velnorm
        goal_angvel = torch.where(condition, zero_state, goal_angvel) # zero goal angvel

        return (
            rb_pos,
            rb_vel,
            rb_ang,
            rb_angvel,
            goal_pos,
            goal_vel,
            goal_velnorm,
            goal_ang,
            goal_angvel,
        )

    def _reset_tasks(
        self,
        reset_buf,
        condition,
        rb_pos,
        rb_vel,
        rb_quats,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        **kwargs,
    ):
        ones = torch.ones_like(reset_buf)

        vel_norm = torch.norm(rb_vel, p=2, dim=-1)
        avel_norm = torch.norm(rb_angvel, p=2, dim=-1)

        reset_buf = torch.where(
            torch.logical_and(condition, (vel_norm > 7)), ones, reset_buf
        )
        reset_buf = torch.where(
            torch.logical_and(condition, (avel_norm > 6.28)), ones, reset_buf
        )
        return reset_buf

class QRYawControlTaskSetObj(TaskSetObj):
    """focus on yaw control in QR mode
    sample:
        - neutral state except yaw angle
    reset:
        - small vel, angvel
    """

    def _sample_state(
        self,
        condition,
        rb_pos,
        rb_vel,
        rb_ang,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        spawn_height,
        **kwargs,
    ):
        zero_state = torch.zeros_like(rb_pos)

        qr_pos = torch.zeros_like(rb_pos)
        qr_pos[..., 2] = spawn_height

        qr_ang = torch.zeros_like(rb_ang)
        qr_ang[..., 2] = rb_ang[..., 2]

        goal_qr_ang = torch.zeros_like(goal_ang)
        goal_qr_ang[..., 2] = goal_ang[..., 2]

        rb_pos = torch.where(condition, qr_pos, rb_pos)  # zero position
        rb_vel = torch.where(condition, zero_state, rb_vel) # zero velocity
        rb_ang = torch.where(condition, qr_ang, rb_ang) # zero roll pitch angle
        rb_angvel = torch.where(condition, zero_state, rb_angvel) # zero angvel

        goal_ang = torch.where(condition, goal_qr_ang, goal_ang) # zero roll pitch goal ang
        goal_vel = torch.where(condition, zero_state, goal_vel) # zero goal vel
        goal_velnorm = torch.where(
            condition[..., 0].unsqueeze(1), torch.zeros_like(goal_velnorm), goal_velnorm
        ) # zero goal velnorm
        goal_angvel = torch.where(condition, zero_state, goal_angvel) # zero goal angvel

        return (
            rb_pos,
            rb_vel,
            rb_ang,
            rb_angvel,
            goal_pos,
            goal_vel,
            goal_velnorm,
            goal_ang,
            goal_angvel,
        )

    def _reset_tasks(
        self,
        reset_buf,
        condition,
        rb_pos,
        rb_vel,
        rb_quats,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        **kwargs,
    ):
        ones = torch.ones_like(reset_buf)

        vel_norm = torch.norm(rb_vel, p=2, dim=-1)
        avel_norm = torch.norm(rb_angvel, p=2, dim=-1)

        reset_buf = torch.where(
            torch.logical_and(condition, (vel_norm > 7)), ones, reset_buf
        )
        reset_buf = torch.where(
            torch.logical_and(condition, (avel_norm > 6.28)), ones, reset_buf
        )
        return reset_buf

class QRAngleControlTaskSetObj(TaskSetObj):
    """focus on angle control in QR mode
    sample:
        - neutral state
        - initial goal and rb angle with less than a 6 degree difference
    reset:
        - angvel
    """

    def _sample_state(
        self,
        condition,
        rb_pos,
        rb_vel,
        rb_ang,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        spawn_height,
        **kwargs,
    ):
        zero_state = torch.zeros_like(rb_pos)

        qr_pos = torch.zeros_like(rb_pos)
        qr_pos[..., 2] = spawn_height

        qr_ang = torch.zeros_like(rb_ang)
        qr_ang[..., 2] = rb_ang[..., 2]

        goal_qr_ang = torch.zeros_like(goal_ang)
        goal_qr_ang[..., 2] = rb_ang + 0.1*torch.rand_like(rb_ang) # 5.7 degree difference

        rb_pos = torch.where(condition, qr_pos, rb_pos)  # zero position
        rb_vel = torch.where(condition, zero_state, rb_vel) # zero velocity
        rb_ang = torch.where(condition, qr_ang, rb_ang) # zero roll pitch angle
        rb_angvel = torch.where(condition, zero_state, rb_angvel) # zero angvel

        goal_ang = torch.where(condition, goal_qr_ang, goal_ang) # zero roll pitch goal ang
        goal_vel = torch.where(condition, zero_state, goal_vel) # zero goal vel
        goal_velnorm = torch.where(
            condition[..., 0].unsqueeze(1), torch.zeros_like(goal_velnorm), goal_velnorm
        ) # zero goal velnorm
        goal_angvel = torch.where(condition, zero_state, goal_angvel) # zero goal angvel

        return (
            rb_pos,
            rb_vel,
            rb_ang,
            rb_angvel,
            goal_pos,
            goal_vel,
            goal_velnorm,
            goal_ang,
            goal_angvel,
        )

    def _reset_tasks(
        self,
        reset_buf,
        condition,
        rb_pos,
        rb_vel,
        rb_quats,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        **kwargs,
    ):
        ones = torch.ones_like(reset_buf)

        avel_norm = torch.norm(rb_angvel, p=2, dim=-1)

        reset_buf = torch.where(
            torch.logical_and(condition, (avel_norm > 6.28)), ones, reset_buf
        )
        return reset_buf


class QRBreakTaskSetObj(TaskSetObj):
    """stabilization task and reduce planar velocity in QR mode
    sample:
        - initial planar velocity
    reset:
        - angvel
    """

    def _sample_state(
        self,
        condition,
        rb_pos,
        rb_vel,
        rb_ang,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        spawn_height,
        vel_lim,
        **kwargs,
    ):
        zero_state = torch.zeros_like(rb_pos)

        qr_pos = torch.zeros_like(rb_pos)
        qr_pos[..., 2] = spawn_height

        rb_vel = sampling(rb_vel.shape, vel_lim, self.device)
        rb_vel[..., 2] = 0

        # qr_ang = sampling(rb_ang.shape, 0.027*torch.pi, self.device)

        rb_pos = torch.where(condition, qr_pos, rb_pos)
        rb_vel = torch.where(condition, zero_state, rb_vel)
        # rb_ang = torch.where(condition, qr_ang, rb_ang)
        rb_angvel = torch.where(condition, zero_state, rb_angvel)

        goal_ang = torch.where(condition, zero_state, goal_ang)
        goal_vel = torch.where(condition, zero_state, goal_vel)
        goal_velnorm = torch.where(
            condition[..., 0].unsqueeze(1), torch.zeros_like(goal_velnorm), goal_velnorm
        )
        goal_angvel = torch.where(condition, zero_state, goal_angvel)

        return (
            rb_pos,
            rb_vel,
            rb_ang,
            rb_angvel,
            goal_pos,
            goal_vel,
            goal_velnorm,
            goal_ang,
            goal_angvel,
        )

    def _reset_tasks(
        self,
        reset_buf,
        condition,
        rb_pos,
        rb_vel,
        rb_quats,
        rb_angvel,
        goal_pos,
        goal_vel,
        goal_velnorm,
        goal_ang,
        goal_angvel,
        **kwargs,
    ):
        ones = torch.ones_like(reset_buf)

        avel_norm = torch.norm(rb_angvel, p=2, dim=-1)

        reset_buf = torch.where(
            torch.logical_and(condition, (avel_norm > 6.28)), ones, reset_buf
        )
        return reset_buf

##########################################################
##############    Smart Task Set Object     ##############
##########################################################

class SmartTaskSet(SmartTask):
    def __init__(self, env_cfg, device) -> None:
        super().__init__(env_cfg, device)
    
        self.sb = env_cfg["task"].get("sampling_bound", 0.75)

        self.taskSets_train = []
        self.taskSets_eval = []

    def reset_tasks(
        self,
        reset_buf,
        task_mode,
        **kwargs,
    ):
        if task_mode == "train":
            for taskSet in self.taskSets_train:
                reset_buf = taskSet.reset_tasks(
                    reset_buf=reset_buf,
                    ids=self.Train.id,
                    **kwargs,
                )
        elif task_mode == "eval":
            for taskSet in self.taskSets_eval:
                reset_buf = taskSet.reset_tasks(
                    reset_buf=reset_buf,
                    ids=self.Train.id,
                    **kwargs,
                )

        return reset_buf

    def sample_robot_state(
        self,
        **kwargs,
    ):
        mode = kwargs["mode"]
        difficulty = kwargs["difficulty"]
        max_difficulty = kwargs["max_difficulty"]

        env_ids = kwargs["env_ids"]
        num_bodies = kwargs["num_bodies"]

        pos_lim = kwargs["pos_lim"]
        vel_lim = kwargs["vel_lim"]
        avel_lim = kwargs["avel_lim"]
        spawn_height = kwargs["spawn_height"]

        goal_pos = kwargs.pop("goal_pos")
        goal_vel = kwargs.pop("goal_vel")
        goal_velnorm = kwargs.pop("goal_velnorm")
        goal_ang = kwargs.pop("goal_ang")
        goal_angvel = kwargs.pop("goal_angvel")

        if mode == "eval":
            difficulty = max_difficulty

        sampling_range = difficulty * self.sb

        rb_pos = sampling(
            (len(env_ids), 3),
            pos_lim * sampling_range,
            self.device,
        )
        rb_pos[..., 2] = spawn_height + sampling(
            len(env_ids),
            spawn_height * sampling_range,
            self.device,
        )
        rb_pos[..., 2] = torch.where(
            rb_pos[..., 2] <= 0.5, spawn_height, rb_pos[..., 2]
        )

        rb_vel = sampling((len(env_ids), 3), vel_lim * sampling_range, self.device)
        rb_ang = sampling((len(env_ids), 3), torch.pi * difficulty, self.device)
        rb_angvel = sampling((len(env_ids), 3), avel_lim * sampling_range, self.device)

        if mode == "train":
            for taskSet in self.taskSets_train:
                (
                    rb_pos,
                    rb_vel,
                    rb_ang,
                    rb_angvel,
                    goal_pos,
                    goal_vel,
                    goal_velnorm,
                    goal_ang,
                    goal_angvel,
                ) = taskSet.sample_robot_state(
                    ids=self.Train.id[env_ids],
                    rb_pos=rb_pos,
                    rb_vel=rb_vel,
                    rb_ang=rb_ang,
                    rb_angvel=rb_angvel,
                    goal_pos=goal_pos,
                    goal_vel=goal_vel,
                    goal_velnorm=goal_velnorm,
                    goal_ang=goal_ang,
                    goal_angvel=goal_angvel,
                    **kwargs,
                )
        elif mode == "eval":
            for taskSet in self.taskSets_eval:
                (
                    rb_pos,
                    rb_vel,
                    rb_ang,
                    rb_angvel,
                    goal_pos,
                    goal_vel,
                    goal_velnorm,
                    goal_ang,
                    goal_angvel,
                ) = taskSet.sample_robot_state(
                    ids=self.Train.id[env_ids],
                    rb_pos=rb_pos,
                    rb_vel=rb_vel,
                    rb_ang=rb_ang,
                    rb_angvel=rb_angvel,
                    goal_pos=goal_pos,
                    goal_vel=goal_vel,
                    goal_velnorm=goal_velnorm,
                    goal_ang=goal_ang,
                    goal_angvel=goal_angvel,
                    **kwargs,
                )

        rb_pos = rb_pos.unsqueeze(1).repeat(1, num_bodies, 1)
        rb_vel = rb_vel.unsqueeze(1).repeat(1, num_bodies, 1)
        rb_ang = rb_ang.unsqueeze(1).repeat(1, num_bodies, 1)
        rb_angvel = rb_angvel.unsqueeze(1).repeat(1, num_bodies, 1)

        return (
            rb_pos,
            rb_vel,
            rb_ang,
            rb_angvel,
            goal_pos,
            goal_vel,
            goal_velnorm,
            goal_ang,
            goal_angvel,
        )

class TailsitterTaskSet(SmartTaskSet):
    def __init__(self, env_cfg, device) -> None:
        super().__init__(env_cfg, device)

        trainSet = env_cfg["task"]["taskSet"]["train"]
        trainSet = re.split(r"\s*,\s*", trainSet)

        evalSet = env_cfg["task"]["taskSet"]["eval"]
        evalSet = re.split(r"\s*,\s*", evalSet)

        if "QRStabilizeSet" in trainSet:
            self.taskSets_train.append(QRStabilizeTaskSetObj("QRStabilizeSet", env_cfg, device))

        if "FWAttSet" in trainSet:
            self.taskSets_train.append(FWAttitudeTaskSetObj("FWAttSet", env_cfg, device))

        if "TranSet" in trainSet:
            self.taskSets_train.append(TransitionTaskSetObj("TranSet", env_cfg, device))

        if "PoseRecoverSet" in trainSet:
            self.taskSets_train.append(PoseRecoverTaskSetObj("PoseRecoverSet", env_cfg, device))

class QuadcopterTaskSet(SmartTaskSet):
    def __init__(self, env_cfg, device) -> None:
        super().__init__(env_cfg, device)

        trainSet = env_cfg["task"]["taskSet"]["train"]
        trainSet = re.split(r"\s*,\s*", trainSet)

        if "QRStabilizeSet" in trainSet:
            self.taskSets_train.append(QRStabilizeTaskSetObj("QRStabilizeSet", env_cfg, device))

        if "PoseRecoverSet" in trainSet:
            self.taskSets_train.append(PoseRecoverTaskSetObj("PoseRecoverSet", env_cfg, device))

        if "QRBreakSet" in trainSet:
            self.taskSets_train.append(QRBreakTaskSetObj("QRBreakSet", env_cfg, device))

        evalSet = env_cfg["task"]["taskSet"]["eval"]
        evalSet = re.split(r"\s*,\s*", evalSet)

        if "QRStabilizeSet" in evalSet:
            self.taskSets_eval.append(QRStabilizeTaskSetObj("QRStabilizeSet", env_cfg, device))

        if "PoseRecoverSet" in evalSet:
            self.taskSets_eval.append(PoseRecoverTaskSetObj("PoseRecoverSet", env_cfg, device))

        if "QRBreakSet" in evalSet:
            self.taskSets_eval.append(QRBreakTaskSetObj("QRBreakSet", env_cfg, device))


class BlimpTaskSet(SmartTaskSet):
    def __init__(self, env_cfg, device) -> None:
        super().__init__(env_cfg, device)

def task_constructor(env_cfg, device):
    if "blimp" in env_cfg["env_name"].lower():
        return BlimpTaskSet(env_cfg, device)
    elif "tailsitter" in env_cfg["env_name"].lower():
        return TailsitterTaskSet(env_cfg, device)
    elif "quadcopter" in env_cfg["env_name"].lower():
        return QuadcopterTaskSet(env_cfg, device)
    else:
        return SmartTaskSet(env_cfg, device)
