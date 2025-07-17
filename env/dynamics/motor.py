import torch
from common.torch_jit_utils import *

def first_order_system(inputState, previousState, sampling_time, time_constant_up, time_constant_down):
    up = _first_order_system_up(inputState, previousState, sampling_time, time_constant_up)
    down = _first_order_system_down(inputState, previousState, sampling_time, time_constant_down)
    outputState = torch.where(inputState>previousState, up, down)
    return outputState

@torch.jit.script
def _first_order_system_up(inputState, previousState, samplingTime, timeConstantUp):
    # type: (Tensor, Tensor, float, Tensor) -> Tensor
    alphaUp = torch.exp(-samplingTime / timeConstantUp)
    outputState = alphaUp * previousState + (1 - alphaUp) * inputState
    return outputState

@torch.jit.script
def _first_order_system_down(inputState, previousState, samplingTime, timeConstantDown):
    # type: (Tensor, Tensor, float, Tensor) -> Tensor
    alphaDown = torch.exp(-samplingTime / timeConstantDown)
    outputState = alphaDown * previousState + (1 - alphaDown) * inputState
    return outputState

def simulate_rotors(rotor_rad, rb_quat, rb_angvel, body_velocity_W, wind_speed_W, 
                    cw, AERO_CONST_MUL_C_T, C_Q, C_M, rotor_drag_coefficient, propeller_inertia, upward, rotor_axes,
                    enable_gyroscopic_torque=True, enable_air_drag=True, enable_rolling_moment=True):
    
    # [N, 4]
    thrusts = _simulate_thrust(rotor_rad, AERO_CONST_MUL_C_T) 

    # [N, 4]
    torques = _simulate_torque(thrusts, cw, C_Q) 

    # [N, 4, 3]
    if enable_gyroscopic_torque:
        body_omega = global_to_local(rb_quat, rb_angvel)
        gyro_torque = _simulate_gyro_torque(rotor_rad, body_omega, propeller_inertia, rotor_axes)
    else:
        gyro_torque = None

    # [N, 3]
    body_velocity_perpendicular = _compute_body_velocity_perpendicular(rb_quat, body_velocity_W, wind_speed_W, upward)

    # [N, 4, 3] the air drag apply to each motor in 3d
    if enable_air_drag:
        air_drag = _simulate_air_drag(rotor_rad, body_velocity_perpendicular, rotor_drag_coefficient)
    else:
        air_drag = None

    # [N, 4, 3] the moment apply to each motor in 3d
    if enable_rolling_moment:
        rolling_moment = _simulate_rolling_moment(rotor_rad, body_velocity_perpendicular, C_M)
    else:
        rolling_moment = None

    return thrusts, torques, air_drag, rolling_moment, gyro_torque


@torch.jit.script
def _simulate_thrust(rotor_speeds, AERO_CONST_MUL_C_T):
    # type: (Tensor, Tensor) -> Tensor
    return AERO_CONST_MUL_C_T * rotor_speeds**2

@torch.jit.script
def _simulate_torque(thrusts, cw, C_Q):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    return -cw * C_Q * thrusts

@torch.jit.script
def _simulate_gyro_torque(rotor_speeds, body_omega, rotor_inertia, rotor_axes):           
    # type: (Tensor, Tensor, float, Tensor) -> Tensor
    rotor_speeds_expanded = rotor_speeds.unsqueeze(-1)        
    rotor_axes_expanded   = rotor_axes.unsqueeze(0)           
    body_omega_expanded   = body_omega.unsqueeze(1)           

    # Rotor velocity vectors in the body frame:
    #   rotor_vel[i,j,:] = rotor_speeds[i,j] * rotor_axes[j,:]
    rotor_velocity = rotor_speeds_expanded * rotor_axes_expanded

    # Gyroscopic torque for each rotor = J_rotor * (omega_rotor x body_omega)
    rotor_gyro_torque = rotor_inertia * torch.cross(rotor_velocity, body_omega_expanded, dim=-1)

    return rotor_gyro_torque

@torch.jit.script
def _compute_body_velocity_perpendicular(rb_quats, body_velocity_W, wind_speed_W, upward):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    upwardI = local_to_global(rb_quats, upward)  # [N,3] <- [N,4], [3]
    relative_wind_velocity_W = body_velocity_W - wind_speed_W # [N, 3]

    ang = (relative_wind_velocity_W * upwardI).sum(1)  # [N] dot product 

    body_velocity_perpendicular = relative_wind_velocity_W - ang[:, None] * upwardI
    return body_velocity_perpendicular

@torch.jit.script
def _simulate_air_drag(rotor_speeds, body_velocity_perpendicular, rotor_drag_coefficient):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    # input : [N, 4], [N, 3], []
    # return : [N, 4, 3]
    return - rotor_drag_coefficient * torch.abs(rotor_speeds[:, :, None]) * body_velocity_perpendicular[:, None, :]

@torch.jit.script
def _simulate_rolling_moment(rotor_speeds, body_velocity_perpendicular, C_M):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    # input : [N, 4], [N, 3], [N, 1]
    # return : [N, 4, 3]
    if C_M.dim()!=0:
        C_M_ = C_M[:, :, None]
    else:
        C_M_ = C_M
    return -C_M_ * torch.abs(rotor_speeds[:, :, None]) * body_velocity_perpendicular[:, None, :]
