import collections.abc
import json
from re import A
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def fix_wandb(d):
    ret = {}
    d = dict(d)
    for k, v in d.items():
        if not "." in k:
            ret[k] = v
        else:
            ks = k.split(".")
            a = fix_wandb({".".join(ks[1:]): v})
            if ks[0] not in ret:
                ret[ks[0]] = {}
            update_dict(ret[ks[0]], a)
    return ret


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionory."""
    if type(val) == dict:
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        print(val)


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# def get_sa_pairs(s: torch.tensor, a: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
#     """s is state, a is action particles
#     pair every state with each action particle

#     for example, 2 samples of state and 3 action particls
#     s = [s0, s1]
#     a = [a0, a1, a2]

#     s_tile = [s0, s1, s0, s1, s0, s1]
#     a_tile = [a0, a1, a2, a0, a1, a2]

#     Args:
#         s (torch.tensor): (number of samples, state dimension)
#         a (torch.tensor): (number of particles, action dimension)

#     Returns:
#         Tuple[torch.tensor, torch.tensor]:
#             s_tile (n_sample*n_particles, state_dim)
#             a_tile (n_sample*n_particles, act_dim)
#     """
#     n_particles = a.shape[0]
#     n_samples = s.shape[0]
#     state_dim = s.shape[1]

#     s_tile = torch.tile(s, (1, n_particles))
#     s_tile = s_tile.reshape(-1, state_dim)

#     a_tile = torch.tile(a, (n_samples, 1))
#     return s_tile, a_tile


# def pile_sa_pairs(
#     s: torch.tensor, a: torch.tensor
# ) -> Tuple[torch.tensor, torch.tensor]:
#     """s is state, a is action particles
#     pair every state with each action particle

#     Args:
#         s (tensor): (number of samples, state dimension)
#         a (tensor): (number of samples, number of particles, action dimension)

#     Returns:
#         Tuple[torch.tensor, torch.tensor]:
#             s_tile (n_sample*n_particles, state_dim)
#             a_tile (n_sample*n_particles, act_dim)
#     """
#     n_samples = s.shape[0]
#     state_dim = s.shape[1]
#     n_particles = a.shape[1]
#     act_dim = a.shape[2]

#     s_tile = torch.tile(s, (1, n_particles))
#     s_tile = s_tile.reshape(-1, state_dim)

#     a_tile = a.reshape(-1, act_dim)
#     return s_tile, a_tile


# def get_sah_pairs(
#     s: torch.tensor, a: torch.tensor, h: torch.tensor
# ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
#     n_samples = a.shape[0]

#     s_tile, a_tile = get_sa_pairs(s, a)
#     h_tile = torch.tile(h, (n_samples, 1))
#     return s_tile, a_tile, h_tile



@torch.jit.script
def get_sw_pairs(
    s: Tensor,  # [N, S]
    w: Tensor   # [T, W]
) -> Tuple[Tensor, Tensor]:
    """
    Produce all (s_i, w_j) pairs for i in [0..N-1], j in [0..T-1].
    Returns N*T rows for each output:

      s_tile: [N*T, S]
      w_tile: [N*T, W]

    Args:
        s: [N, S] - states or items to be paired
        w: [T, W] - tasks or other items to be paired

    Returns:
        s_tile, w_tile
    """
    n_samples = s.shape[0]  # N
    n_w = w.shape[0]        # T

    # 1) Replicate every row of s T times => [N*T, S]
    s_tile = s.repeat(n_w, 1)

    # 2) Replicate every row of w N times => [N*T, W]
    w_tile = w.repeat_interleave(n_samples, dim=0)

    return s_tile, w_tile


@torch.jit.script
def get_sasdw_pairs(
    s: Tensor,      # [N, S]
    a: Tensor,      # [N, A]
    snext: Tensor,  # [N, S]
    done: Tensor,   # [N, 1]
    w: Tensor       # [T, W]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    For each of the N (s,a,s_next,done) tuples, pair them with every task in w.
    The result is N*T rows for each output:

      s_tile:     [N*T, S]
      a_tile:     [N*T, A]
      snext_tile: [N*T, S]
      done_tile:  [N*T, 1]
      w_tile:     [N*T, W]

    Args:
        s:     [N, S]  - states
        a:     [N, A]  - actions
        snext: [N, S]  - next states
        done:  [N, 1]  - done flags
        w:     [T, W]  - tasks

    Returns:
        (s_tile, a_tile, snext_tile, done_tile, w_tile)
    """
    n_samples = s.shape[0]  # N
    n_w = w.shape[0]        # T
    s_dim = s.shape[1]
    a_dim = a.shape[1]
    done_dim = done.shape[1]

    # Repeat each row in (s,a,snext,done) horizontally T times, then reshape
    s_tile = s.repeat(1, n_w).reshape(-1, s_dim)       # [N*T, S]
    a_tile = a.repeat(1, n_w).reshape(-1, a_dim)       # [N*T, A]
    snext_tile = snext.repeat(1, n_w).reshape(-1, s_dim)  # [N*T, S]
    done_tile = done.repeat(1, n_w).reshape(-1, done_dim) # [N*T, 1]

    # Repeat w vertically N times so each row of (s_tile,a_tile,...) sees every w
    w_tile = w.repeat(n_samples, 1)  # [N*T, W]

    return s_tile, a_tile, snext_tile, done_tile, w_tile


@torch.jit.script
def get_sasdgw_pairs(
    s: Tensor,  # [N, S]
    a: Tensor,  # [N, A]
    snext: Tensor,  # [N, S]
    done: Tensor,  # [N, 1]
    g: Tensor,  # [N, G]
    w: Tensor,  # [T, W]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Produce all combinations (Cartesian product) of:
      - N experience tuples:  (s, a, s_next, done)
      - N goals:              g
      - T tasks:              w

    Resulting shape for each output is [N*N*T, ...]:
      s_tile:     [N*N*T, S]
      a_tile:     [N*N*T, A]
      snext_tile: [N*N*T, S]
      done_tile:  [N*N*T, 1]
      g_tile:     [N*N*T, G]
      w_tile:     [N*N*T, W]
    """

    N = s.shape[0]  # number of (s,a,s_next,done) tuples and also number of goals
    T = w.shape[0]  # number of tasks

    # -------------------------------------------------------
    # 1) Tile (s, a, snext, done) over all N goals, then over T tasks
    # -------------------------------------------------------
    # First, repeat_interleave(N, dim=0) → expands dimension from [N, *] to [N*N, *]
    #    so each row i is duplicated for each goal j in [0..N-1].
    # Then, repeat(..., 1) by T times → [N*N*T, *], giving every pair (i,j) for each task k.

    s_tile = s.repeat_interleave(N, dim=0)  # [N*N, S]
    s_tile = s_tile.repeat(T, 1)  # [N*N*T, S]

    a_tile = a.repeat_interleave(N, dim=0)  # [N*N, A]
    a_tile = a_tile.repeat(T, 1)  # [N*N*T, A]

    snext_tile = snext.repeat_interleave(N, dim=0)  # [N*N, S]
    snext_tile = snext_tile.repeat(T, 1)  # [N*N*T, S]

    done_tile = done.repeat_interleave(N, dim=0)  # [N*N, 1]
    done_tile = done_tile.repeat(T, 1)  # [N*N*T, 1]

    # -------------------------------------------------------
    # 2) Tile goals g over states:
    #    - repeat(N, 1) → [N*N, G]
    #    - then repeat for T tasks → [N*N*T, G]
    # -------------------------------------------------------
    g_tile = g.repeat(N, 1)  # [N*N, G]
    g_tile = g_tile.repeat(T, 1)  # [N*N*T, G]

    # -------------------------------------------------------
    # 3) Finally, tile the tasks w for all pairs (i, j).
    #    We have (N*N) pairs, so we repeat that block → [N*N*T, W]
    # -------------------------------------------------------
    w_tile = w.repeat(N * N, 1)  # [N*N*T, W]

    return s_tile, a_tile, snext_tile, done_tile, g_tile, w_tile


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


def grad_true(network):
    for param in network.parameters():
        param.requires_grad = True


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape
    assert len(tensor_shape) == len(
        expected_shape
    ), f"expect len a {len(tensor_shape)}, b {len(expected_shape)}"
    assert all(
        [a == b for a, b in zip(tensor_shape, expected_shape)][1:]
    ), f"expect shape a {tensor_shape}, b {expected_shape}"


def np2ts(obj: np.ndarray) -> torch.Tensor:
    if isinstance(obj, np.ndarray) or isinstance(obj, float):
        obj = torch.tensor(obj, dtype=torch.float32).to(device)
    return obj


def ts2np(obj: torch.Tensor) -> np.ndarray:
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu().detach().numpy()
    return obj


def check_samples(obj):
    if obj.ndim > 1:
        n_samples = obj.shape[0]
    else:
        n_samples = 1
    return n_samples


def check_obs(obs, obs_dim):
    obs = np2ts(obs)
    n_samples = check_samples(obs)
    obs = obs.reshape(n_samples, obs_dim)
    return obs


def check_act(action, action_dim, type=np.float32):
    # action = ts2np(action)
    # n_samples = check_samples(action)
    # action = action.reshape(n_samples, action_dim).astype(type)
    return torch.clamp(action, min=-1, max=1)


def to_batch(
    state,
    feature,
    action,
    reward,
    next_state,
    done,
    device,
):
    state = torch.FloatTensor(state).to(device)
    feature = torch.FloatTensor(feature).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    done = torch.FloatTensor(done).to(device)
    return state, feature, action, reward, next_state, done


def to_batch_rnn(
    state,
    feature,
    action,
    reward,
    next_state,
    done,
    h_in0,
    h_out0,
    h_in1,
    h_out1,
    device,
):
    state, feature, action, reward, next_state, done = to_batch(
        state, feature, action, reward, next_state, done, device
    )
    return (
        state,
        feature,
        action,
        reward,
        next_state,
        done,
        h_in0,
        h_out0,
        h_in1,
        h_out1,
    )


def update_params(
    optim, network, loss, grad_clip=None, retain_graph=False, set_to_none=True
):
    optim.zero_grad(set_to_none)
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optim.step()


def update_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def linear_schedule(cur_step, tot_step, schedule):
    progress = cur_step / tot_step
    return np.clip(
        progress * (schedule[1] - schedule[0]) + schedule[0],
        np.min(np.array(schedule)),
        np.max(np.array(schedule)),
    )


def dump_cfg(path, obj):
    with open(path, "w") as fp:
        json.dump(dict(obj), fp, default=lambda o: o.__dict__, indent=4, sort_keys=True)
