import torch
from common.torch_jit_utils import *

@torch.jit.script
def simulate_wind(rb_quats, wind_speed_W, body_area):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    k = 0.67375 # rho*cd/2, rho: air density 1.225, cd: drag coeff 1.1
    wind_speed_B = global_to_local(rb_quats, wind_speed_W)
    return k*body_area*wind_speed_B**2

# @torch.jit.script
def simulate_aerodynamics(
    rb_quats,
    rb_vel,
    wind,
    alpha0,
    cla,
    cda,
    alphaStall,
    claStall,
    cdaStall,
    forward,
    upward,
    area,
    rho,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    vel = rb_vel + wind.unsqueeze(1)

    forwardI = local_to_global(rb_quats, forward)  # [N,B,4], [B,3]
    upwardI = local_to_global(rb_quats, upward)  # [N,B,4], [B,3]

    ldNormal = torch.cross(forwardI, upwardI, dim=2)  # [N,B,3]
    ldNormal = torch.nn.functional.normalize(ldNormal, p=2.0, dim=2)  # [N,B,3]

    # check sweep (angle between vel and lift-drag-plane)
    sinSweepAngle = (ldNormal * vel).sum(2)  # dot product [N,B]
    sinSweepAngle = sinSweepAngle / vel.norm(p=2, dim=2)  # [N,B] <- [N,B] / [N,B]

    # get cos from trig identity
    cosSweepAngle2 = 1.0 - sinSweepAngle**2  # [N,B]
    sweep = torch.asin(sinSweepAngle)  # [N,B]

    # truncate to within +/-90 deg
    sweep = torch.where(sweep > 0.5 * torch.pi, sweep - torch.pi, sweep)
    sweep = torch.where(sweep < -0.5 * torch.pi, sweep + torch.pi, sweep)

    velInLDPlane = torch.cross(vel, ldNormal, dim=2)  # [N,B,3]
    velInLDPlane = torch.cross(ldNormal, velInLDPlane, dim=2)  # [N,B,3]

    dragDirection = -velInLDPlane
    dragDirection = torch.nn.functional.normalize(
        dragDirection, p=2.0, dim=2
    )  # [N,B,3]

    liftDirection = torch.cross(ldNormal, velInLDPlane, dim=2)  # [N,B,3]
    liftDirection = torch.nn.functional.normalize(
        liftDirection, p=2.0, dim=2
    )  # [N,B,3]

    cosAlpha = (forwardI * velInLDPlane).sum(2)  # dot product [N,B]
    cosAlpha = cosAlpha / (
        forwardI.norm(p=2, dim=2) * velInLDPlane.norm(p=2, dim=2)
    )  # [N,B]
    cosAlpha = torch.clamp(cosAlpha, -1.0, 1.0)

    alphaSign = (upwardI * velInLDPlane).sum(2)  # dot product [N,B]
    alphaSign = -alphaSign / (
        upwardI.norm(p=2, dim=2) * velInLDPlane.norm(p=2, dim=2)
    )  # [N,B]

    alpha = torch.where(
        alphaSign > 0.0, alpha0 + torch.acos(cosAlpha), alpha0 - torch.acos(cosAlpha)
    )  # [N, B]

    # truncate to within +/-90 deg
    alpha = torch.where(alpha > 0.5 * torch.pi, alpha - torch.pi, alpha)  # [N, B]
    alpha = torch.where(alpha < -0.5 * torch.pi, alpha + torch.pi, alpha)

    speedInLDPlane = velInLDPlane.norm(p=2, dim=2)  # [N, B]
    q = 0.5 * rho * speedInLDPlane**2  # [N, B]

    cl = cla * alpha * cosSweepAngle2
    tmp = (cla * alphaStall + claStall * (alpha - alphaStall)) * cosSweepAngle2
    tmp = torch.max(torch.tensor(0), tmp)
    cl = torch.where(alpha > alphaStall, tmp, cl)
    tmp = (-cla * alphaStall + claStall * (alpha + alphaStall)) * cosSweepAngle2
    tmp = torch.min(torch.tensor(0), tmp)
    cl = torch.where(alpha < -alphaStall, tmp, cl)  # [N, B]

    lift = torch.einsum(
        "ij,ij,j,ijk->ijk", cl, q, area, liftDirection
    )  # [N, B, 3] <- [N, B], [N, B], [B], [N, B, 3]

    cd = cda * alpha * cosSweepAngle2
    tmp = (cda * alphaStall + cdaStall * (alpha - alphaStall)) * cosSweepAngle2
    cd = torch.where(alpha > alphaStall, tmp, cd)
    tmp = (-cda * alphaStall + cdaStall * (alpha + alphaStall)) * cosSweepAngle2
    cd = torch.where(alpha < -alphaStall, tmp, cd)

    cd = torch.abs(cd)
    drag = torch.einsum(
        "ij,ij,j,ijk->ijk", cd, q, area, dragDirection
    )  # [N, B, 3] <- [N, B], [N, B], [B], [N, B, 3]

    forces = lift + drag  # gloabal frame
    # return quaternion_apply(quaternion_invert(rb_quats), forces)  # convert to local frame
    return global_to_local(rb_quats, forces)








