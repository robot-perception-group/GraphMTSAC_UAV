import isaacgym

import hydra
from omegaconf import DictConfig, OmegaConf
from common.util import omegaconf_to_dict, print_dict, fix_wandb, update_dict


import torch
import numpy as np

import wandb


def get_agent(cfg_dict):

    if "rmamtsac" in cfg_dict["agent"]["name"].lower():
        from agents.rmamtsac import RMAMultitaskSACAgent

        agent = RMAMultitaskSACAgent(cfg_dict)

    elif "mtsac" in cfg_dict["agent"]["name"].lower():
        from agents.mtsac import MultitaskSACAgent

        agent = MultitaskSACAgent(cfg_dict)

    elif "sac" in cfg_dict["agent"]["name"].lower():
        from agents.sac import SACAgent

        agent = SACAgent(cfg_dict)

    else:
        raise NotImplementedError

    return agent


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    if cfg_dict["wandb_log"]:
        wandb.init()
    else:
        wandb.init(mode="disabled")

    wandb_dict = fix_wandb(wandb.config)
    update_dict(cfg_dict, wandb_dict)
    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    print_dict(cfg_dict)

    torch.manual_seed(cfg_dict["seed"])
    np.random.seed(cfg_dict["seed"])

    agent = get_agent(cfg_dict)

    agent.run()
    wandb.finish()


if __name__ == "__main__":
    launch_rlg_hydra()
