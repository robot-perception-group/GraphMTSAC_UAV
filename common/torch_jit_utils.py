import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Union, List, Tuple

torch.pi = math.pi
torch.arctan2 = torch.atan2

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def copysignMulti(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape)
    return torch.abs(a) * torch.sign(b)

@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)

@torch.jit.script
def _sampling_int(size, scale, device):
    # type: (int, float, str) -> Tensor
    return scale * 2 * (torch.rand(size, device=device) - 0.5)

@torch.jit.script
def _sampling_int_tensor(size, scale, device):
    # type: (int, Tensor, str) -> Tensor
    return scale * 2 * (torch.rand(size, device=device) - 0.5)

@torch.jit.script
def _sampling_tuple(size, scale, device):
    # type: (Tuple[int, int], float, str) -> Tensor
    return scale * 2 * (torch.rand(size, device=device) - 0.5)

@torch.jit.script
def _sampling_tuple_tensor(size, scale, device):
    # type: (Tuple[int, int], Tensor, str) -> Tensor
    return scale * 2 * (torch.rand(size, device=device) - 0.5)


def sampling(size: Union[int, Tuple], scale: Union[float, torch.Tensor], device: str):
    if isinstance(size, int):
        if isinstance(scale, float):
            return _sampling_int(size, scale, device)
        else:
            return _sampling_int_tensor(size, scale, device)
    elif isinstance(size, Tuple):
        if isinstance(scale, float):
            return _sampling_tuple(size, scale, device)
        else:
            return _sampling_tuple_tensor(size, scale, device)



@torch.jit.script
def _sample_from_range(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower

@torch.jit.script
def _sample_from_range_int(lower, upper, shape, device):
    # type: (float, float, int, str) -> Tensor
    return (upper - lower) * torch.rand(shape, device=device) + lower


def sample_from_range(lower, upper, shape, device):
    if isinstance(shape, int):
        return _sample_from_range_int(lower, upper, shape, device)
    
    elif isinstance(shape, Tuple):
        return _sample_from_range(lower, upper, shape, device)

##########################################
################ Rotation ################
##########################################

def euler_from_quat(q, order=[0,1,2,3]):
    return _euler_from_quat(q, order)

@torch.jit.script
def _euler_from_quat(q, order):
    # type: (Tensor, List[int]) -> Tensor

    qx, qy, qz, qw = order
    
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (
        q[:, qw] * q[:, qw]
        - q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        + q[:, qz] * q[:, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp)
    )

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (
        q[:, qw] * q[:, qw]
        + q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        - q[:, qz] * q[:, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)], dim=-1)

@torch.jit.script
def euler_from_quat_multi(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, :, qw] * q[:, :, qx] + q[:, :, qy] * q[:, :, qz])
    cosr_cosp = (
        q[:, :, qw] * q[:, :, qw]
        - q[:, :, qx] * q[:, :, qx]
        - q[:, :, qy] * q[:, :, qy]
        + q[:, :, qz] * q[:, :, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, :, qw] * q[:, :, qy] - q[:, :, qz] * q[:, :, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysignMulti(np.pi / 2.0, sinp), torch.asin(sinp)
    )

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, :, qw] * q[:, :, qz] + q[:, :, qx] * q[:, :, qy])
    cosy_cosp = (
        q[:, :, qw] * q[:, :, qw]
        + q[:, :, qx] * q[:, :, qx]
        - q[:, :, qy] * q[:, :, qy]
        - q[:, :, qz] * q[:, :, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)

@torch.jit.script
def quat_from_euler(euler):
    roll, pitch, yaw = euler[...,0], euler[...,1], euler[...,2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)

@torch.jit.script
def localToGlobalRot(euler, vec):
    roll, pitch, yaw = euler[...,0], euler[...,1], euler[...,2]
    x, y, z = vec[...,0], vec[...,1], vec[...,2]

    cosa = torch.cos(yaw)
    sina = torch.sin(yaw)

    cosb = torch.cos(pitch)
    sinb = torch.sin(pitch)

    cosy = torch.cos(roll)
    siny = torch.sin(roll)

    xp = (
        x * cosa * cosb
        + y * (cosa * sinb * siny - sina * cosy)
        + z * (cosa * sinb * cosy + sina * siny)
    )
    yp = (
        x * sina * cosb
        + y * (sina * sinb * siny + cosa * cosy)
        + z * (sina * sinb * cosy - cosa * siny)
    )
    zp = -x * sinb + y * cosb * siny + z * cosb * cosy
    return torch.stack([xp, yp, zp], dim=-1)

@torch.jit.script
def globalToLocalRot(euler, vec):
    roll, pitch, yaw = euler[...,0], euler[...,1], euler[...,2]
    x, y, z = vec[...,0], vec[...,1], vec[...,2]

    cosa = torch.cos(yaw)
    sina = torch.sin(yaw)

    cosb = torch.cos(pitch)
    sinb = torch.sin(pitch)

    cosy = torch.cos(roll)
    siny = torch.sin(roll)

    xp = x * cosa * cosb + y * sina * cosb - z * sinb
    yp = (
        x * (cosa * sinb * siny - sina * cosy)
        + y * (sina * sinb * siny + cosa * cosy)
        + z * cosb * siny
    )
    zp = (
        x * (cosa * sinb * cosy + sina * siny)
        + y * (sina * sinb * cosy - cosa * siny)
        + z * cosb * cosy
    )
    return torch.stack([xp, yp, zp], dim=-1)

@torch.jit.script
def globalToLocalRot2(roll, pitch, yaw, x, y, z):
    cosa = torch.cos(yaw)
    sina = torch.sin(yaw)

    cosb = torch.cos(pitch)
    sinb = torch.sin(pitch)

    cosy = torch.cos(roll)
    siny = torch.sin(roll)

    xp = x * cosa * cosb + y * sina * cosb - z * sinb
    yp = (
        x * (cosa * sinb * siny - sina * cosy)
        + y * (sina * sinb * siny + cosa * cosy)
        + z * cosb * siny
    )
    zp = (
        x * (cosa * sinb * cosy + sina * siny)
        + y * (sina * sinb * cosy - cosa * siny)
        + z * cosb * cosy
    )
    return xp, yp, zp

@torch.jit.script
def check_angle(ang): # map angle to [-pi, pi]
    ang %= 2*torch.pi
    ang = torch.where(ang > torch.pi, ang - 2 * torch.pi, ang)
    ang = torch.where(ang < -torch.pi, ang + 2 * torch.pi, ang)
    return ang

@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
        * 2.0
    )
    return a + b + c

@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

@torch.jit.script
def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part last.
        b: Quaternions as tensor of shape (..., 4), real part last.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    ax, ay, az, aw = torch.unbind(a, -1)
    bx, by, bz, bw = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ox, oy, oz, ow), -1)

@torch.jit.script
def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            last, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([-1, -1, -1, 1], device=quaternion.device)
    return quaternion * scaling

@torch.jit.script
def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part last, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((point, real_parts), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., :-1]

@torch.jit.script
def quat_rotate2(q, v): #[N,B,4], [B,3]
    shape = q.shape
    v=v.unsqueeze(0).repeat(shape[0],1,1) #[N,B,3]
    q_w = q[..., -1] # [N, B]
    q_vec = q[..., :3] # [N, B, 3]
    tmp = (2.0 * q_w**2 - 1.0) #[N,B]
    a = torch.einsum('ij,ijk->ijk',tmp, v) # [N,B,3] <- [N,B], [N,B,3]
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    tmp = torch.einsum('ijk,ijk->ij',q_vec, v) #[N,B]
    c = 2*torch.einsum('ijk,ij->ijk',q_vec,tmp)
    return a + b + c

@torch.jit.script
def global_to_local(q,v):
    return quaternion_apply(quaternion_invert(q), v)

@torch.jit.script
def local_to_global(q,v):
    return quaternion_apply(q, v)

@torch.jit.script
def angle_between_two_vectors(v1, v2):
    dot_prod = torch.einsum('ij,ij->i', v1, v2)
    v1_norm = v1.norm(p=2,dim=-1)
    v2_norm = v2.norm(p=2,dim=-1)
    tmp = dot_prod / (v1_norm*v2_norm+1e-6)
    tmp = torch.clip(tmp, -1, 1)
    ang = torch.acos(tmp)
    return ang

@torch.jit.script
def quat_rot_of_two_vector(v1, v2):
    vn = torch.cross(v1, v2, dim=-1)

    cos_ang = torch.sum(v1*v2, dim=-1)/(v1.norm(p=2, dim=-1)*v2.norm(p=2, dim=-1))
    ang = torch.acos(cos_ang) % (2 * math.pi)
    
    half = ang/2
    w = torch.cos(half)
    sinhalf = torch.sin(half)
    
    x = sinhalf[:, None]*vn

    q = torch.stack([x[...,0], x[...,1], x[...,2], w], dim=-1)

    return q/q.norm(p=1,dim=-1)

@torch.jit.script
def compute_heading(yaw, rel_pos):
    # type: (Tensor, Tensor) -> Tensor
    return torch.arctan2(rel_pos[:, 1], rel_pos[:, 0]) - torch.pi - yaw

@torch.jit.script
def heading(rb_rot, rel_pos, forward):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    forwardI = local_to_global(rb_rot, forward)
    angle = check_angle(angle_between_two_vectors(forwardI, rel_pos))
    return angle
