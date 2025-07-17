from env import env_map
from common.task.feature import feature_constructor
from common.task.taskset import task_constructor

def env_constructor(env_cfg, device):
    env = env_map[env_cfg["env_name"]](env_cfg)
    feature = feature_constructor(env_cfg, device)
    # task = task_constructor(env_cfg, device)
    # return env, feature, task
    return env, feature

