import isaacgym

from omegaconf import DictConfig, OmegaConf
import argparse
import json
from common.util import (
    print_dict,
    AverageMeter,
)
import math
from run import get_agent

import torch
import numpy as np
from common.torch_jit_utils import *
from tkinter import *

import torch.nn.utils.prune as prune

with_control_panel = False
steps = 500


class PlayUI:
    def __init__(self, cfg_dict, model_path) -> None:
        if with_control_panel:
            self.root = Tk()
            self.root.title("test")
            self.root.geometry("1300x200")
            self.frame = Frame(self.root)
            self.frame.pack()

        # init and load agent
        self.agent = get_agent(cfg_dict)
        self.agent.load_torch_model(model_path)

        self.weights = self.agent.env.task.Eval.W.clone()
        self.weightLabels = cfg_dict["env"]["feature"]["names"]

        if with_control_panel:
            self.generate_scales()
            self.print_rb_info()

    def weight_update_function(self, dimension):
        def update_val(val):
            self.weights[..., dimension] = float(val)
            self.agent.env.task.Eval.update_task(self.weights.clone())
            print("update task:", self.weights)

        return update_val

    def target_update_function(self, dimension):
        def update_val(val):
            self.agent.env.wp.ang[..., 2] = float(val)

        return update_val

    def add_scale(self, dimension, gen_func, label, range=(0, 1), type="weight"):
        scale = Scale(
            self.frame,
            from_=range[0],
            to=range[1],
            digits=3,
            resolution=0.01,
            label=label,
            orient=VERTICAL,
            command=gen_func(dimension),
        )
        if type == "weight":
            scale.set(self.agent.env.task.Eval.W[0, dimension].item())
        scale.pack(side=LEFT)

    def generate_scales(self):
        for i, label in enumerate(self.weightLabels):
            self.add_scale(
                dimension=i, gen_func=self.weight_update_function, label=label
            )

        self.add_scale(
            dimension=3,
            gen_func=self.target_update_function,
            label="target yaw",
            range=(-np.pi, np.pi),
            type="target",
        )

    def print_rb_info(self):
        self.rew = DoubleVar(name="reward")  # instantiate the IntVar variable class
        self.rew.set(0.0)  # set it to 0 as the initial value

        self.act0 = DoubleVar(name="act0")  # instantiate the IntVar variable class
        self.act0.set(0.0)  # set it to 0 as the initial value

        self.act1 = DoubleVar(name="act1")  # instantiate the IntVar variable class
        self.act1.set(0.0)  # set it to 0 as the initial value

        self.act2 = DoubleVar(name="act2")  # instantiate the IntVar variable class
        self.act2.set(0.0)  # set it to 0 as the initial value

        # the label's textvariable is set to the variable class instance
        Label(self.root, text="step reward: ").pack(side=LEFT)
        Label(self.root, textvariable=self.rew).pack(side=LEFT)

        Label(self.root, text="act0: ").pack(side=LEFT)
        Label(self.root, textvariable=self.act0).pack(side=LEFT)

        Label(self.root, text="act1: ").pack(side=LEFT)
        Label(self.root, textvariable=self.act1).pack(side=LEFT)

        Label(self.root, text="act2: ").pack(side=LEFT)
        Label(self.root, textvariable=self.act2).pack(side=LEFT)

    def _debug_ui(self):
        # only runs UI loop without inference
        while True:
            self.root.update_idletasks()
            self.root.update()

    def play(self):
        print("task.Eval:", self.agent.env.task.Eval.W)
        avgStepRew = AverageMeter(1, 20).to(self.agent.device)
        rec_r, rec_roll, rec_pitch, rec_yaw = [], [], [], []

        s = self.agent.reset_env()
        for _ in range(steps):
            if with_control_panel:
                self.root.update_idletasks()
                self.root.update()

            a = self.agent.act(s, self.agent.env.task.Eval, "exploit")

            self.agent.env.step(a)
            s_next = self.agent.env.obs_buf.clone()
            # self.agent.env.reset()

            r = self.agent.calc_reward(s_next, a, self.agent.env.task.Eval.W)
            avgStepRew.update(r)
            s = s_next

            if with_control_panel and self.rew:
                self.rew.set(avgStepRew.get_mean())
                self.act0.set(a[0, 0].item())
                self.act1.set(a[0, 1].item())
                self.act2.set(a[0, 2].item())

            rec_r.append(r.mean())

            euler = check_angle((s[:, 0:3] * np.pi))
            rec_roll.append(euler[:, 0].mean())
            rec_pitch.append(euler[:, 1].mean())
            rec_yaw.append(euler[:, 2].mean())

            print("euler", euler, "actions:", a)

        rec_r = torch.stack(rec_r).squeeze().cpu().numpy()
        rec_roll = torch.stack(rec_roll).squeeze().cpu().numpy()
        rec_pitch = torch.stack(rec_pitch).squeeze().cpu().numpy()
        rec_yaw = torch.stack(rec_yaw).squeeze().cpu().numpy()

        return rec_r, rec_roll, rec_pitch, rec_yaw


def modify_cfg(cfg_dict):

    # change these
    cfg_dict["agent"]["phase"] = 1
    cfg_dict["agent"]["save_model"] = False
    cfg_dict["agent"]["policy_net_kwargs"]["compile_method"] = None

    cfg_dict["env"]["num_envs"] = 1

    cfg_dict["env"]["dynamics"]["add_ground"] = False
    cfg_dict["env"]["dynamics"]["enable_wind"] = False
    cfg_dict["env"]["dynamics"]["wind_to_velocity_ratio"] = [0, 0, 0]
    cfg_dict["env"]["dynamics"]["domain_rand"] = True

    cfg_dict["env"]["task"]["taskSet"]["eval"] = "AttSet"
    cfg_dict["env"]["task"]["taskSet"]["AttSet"] = [[1,1,1,1,.3,.3,.3]] # [1,1,1,1,.3,.3,.3],[1,0,0,0,0,0,0]

    cfg_dict["env"]["task"]["task_mode"] = "eval"
    cfg_dict["env"]["task"]["difficulty"] = 0.2
    cfg_dict["env"]["task"]["rand_rb_pos"] = False
    cfg_dict["env"]["task"]["rand_rb_vel"] = False
    cfg_dict["env"]["task"]["rand_rb_ang"] = False
    cfg_dict["env"]["task"]["rand_rb_avel"] = False
    cfg_dict["env"]["task"]["rand_rb_actuator"] = False
    cfg_dict["env"]["task"]["spawn_height"] = 5
    cfg_dict["env"]["task"]["init_pos"] = [0, 0, 0]
    cfg_dict["env"]["task"]["init_vel"] = [0, 0, 0]
    cfg_dict["env"]["task"]["init_ang"] = [0, 0, 0]
    cfg_dict["env"]["task"]["init_avel"] = [0, 0, 0]

    cfg_dict["env"]["goal"]["goal_type"] = "fix"  # [fix, random]
    cfg_dict["env"]["goal"]["position_target_type"] = "center"  # [waypoints, center]
    cfg_dict["env"]["goal"]["wps_shape"] = "square"  # square, hourglass, circle
    cfg_dict["env"]["goal"]["trigger_dist"] = 1
    cfg_dict["env"]["goal"]["rand_vel"] = False
    cfg_dict["env"]["goal"]["rand_ang"] = False
    cfg_dict["env"]["goal"]["rand_angvel"] = False
    cfg_dict["env"]["goal"]["target_ang"] = [0, 0, -np.pi / 2]
    
    # don't change these
    cfg_dict["env"]["goal"]["visualize_goal"] = True
    cfg_dict["env"]["dynamics"]["visualize_force"] = True
    # cfg_dict["env"]["dynamics"]["visualize_aeroforce"] = False
    cfg_dict["env"]["mode"] = "play"
    cfg_dict["env"]["sim"]["headless"] = False

    cfg_dict["agent"]["curriculum"] = False
    cfg_dict["agent"]["load_model"] = False

    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    cfg_dict["buffer"]["min_n_experience"] = 0

    print_dict(cfg_dict)

    return cfg_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    seed = 999
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_folder = "logs/rmamtsac/Quadcopter/2025-02-24-12-38-46"
    model_checkpoint = "phase1_best"
    plot = True

    cfg_path = model_folder + "/cfg"
    model_path = model_folder + "/" + model_checkpoint + "/"

    cfg_dict = None
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    cfg_dict = modify_cfg(cfg_dict)

    playob = PlayUI(cfg_dict, model_path)
    reward, roll, pitch, yaw = playob.play()

    if plot:
        xs = [x for x in range(len(reward))]

        fig, axs = plt.subplots(4)
        fig.suptitle("performance")
        axs[0].plot(xs, reward)
        axs[0].set_ylabel("reward")
        axs[0].set_ylim([0, 1.5])
        axs[0].grid()

        axs[1].plot(xs, roll)
        axs[1].set_ylabel("roll")
        axs[1].set_ylim([-np.pi, np.pi])
        axs[1].grid()

        axs[2].plot(xs, pitch)
        axs[2].set_ylabel("pitch")
        axs[2].set_ylim([-np.pi, np.pi])
        axs[2].grid()

        axs[3].plot(xs, yaw)
        axs[3].set_ylabel("yaw")
        axs[3].set_ylim([-np.pi, np.pi])
        axs[3].grid()

        target = cfg_dict["env"]["goal"]["target_ang"]
        axs[1].plot(xs, [target[0]] * len(xs))
        axs[2].plot(xs, [target[1]] * len(xs))
        axs[3].plot(xs, [target[2]] * len(xs))

        plt.show()
        plt.savefig(f"pose_recovery_performance_{seed}.png")
        plt.close()