import torch
from common.torch_jit_utils import *


@torch.jit.script
def PID(err, err_sum, err_prev, gain, pid_param, delta_t):
    # type: (Tensor, Tensor, Tensor, float, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]

    err_sum += err * delta_t
    err_sum = torch.clip(err_sum, -1, 1)
    err_i = err_sum

    err_d = (err - err_prev) / (delta_t)
    err_prev = err

    pid = torch.concat([err, err_i, err_d], dim=1)
    ctrl = gain * (pid @ pid_param)
    return ctrl, err_sum, err_prev


class PIDController:
    def __init__(
        self,
        pid_param,
        gain,
        device,
        offset=0.0,
        delta_t=0.01,
        input_clip=[-1.0, 1.0],
    ):
        self.device = device
        self.pid_param = torch.tensor(pid_param, device=device)[:, None]
        self.gain = gain
        self.offset = offset
        self.delta_t = delta_t
        self.input_clip = input_clip

        self.initialized = False

    def action(self, err):
        err = torch.clip(err, self.input_clip[0], self.input_clip[1])
        if err.dim() == 1:
            err = err[:, None]

        if not self.initialized:
            self.err_sum = torch.zeros_like(err)
            self.err_prev = torch.zeros_like(err)
            self.initialized = True

        ctrl, self.err_sum, self.err_prev = PID(
            err, self.err_sum, self.err_prev, self.gain, self.pid_param, self.delta_t
        )
        return ctrl + self.offset

    def clear(self):
        if self.initialized:
            self.err_sum = torch.zeros_like(self.err_sum)
            self.err_prev = torch.zeros_like(self.err_sum)


class TailsitterQRAttitudeControl:
    ctrl_cfg = {
        "roll": {
            "pid_param": torch.tensor([1.0, 0.05, 0.5]),
            "gain": 0.02,
            "input_clip": [-0.1, 0.1],
        },
        "pitch": {
            "pid_param": torch.tensor([1.0, 0.05, 0.5]),
            "gain": 0.03,
            "input_clip": [-0.1, 0.1],
        },
        "yaw": {
            "pid_param": torch.tensor([1.0, 0.05, 0.5]),
            "gain": 0.01,
            "input_clip": [-0.1, 0.1],
        },
        "alt": {
            "pid_param": torch.tensor([1.0, 0.05, 0]),
            "gain": 0.02,
            "offset": 0.33,
        },
    }

    def __init__(self, device):
        delta_t = 0.01

        self.roll_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["roll"],
        )
        self.pitch_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["pitch"],
        )
        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )

        self.slice_error_pos = slice(0, 0 + 3)
        self.slice_robot_quat = slice(3, 3 + 4)
        self.slice_goal_ang = slice(13, 13 + 3)

    def act(self, s):
        z, r, p, y = self.parse_observation(s)

        thrust = self.alt_ctrl.action(z)
        roll_ctrl = self.roll_ctrl.action(r)
        pitch_ctrl = self.pitch_ctrl.action(p)
        yaw_ctrl = self.yaw_ctrl.action(-y)

        return torch.clip(
            torch.concat([thrust, roll_ctrl, pitch_ctrl, yaw_ctrl], dim=1), -1, 1
        )

    def parse_observation(self, s):
        error_pos = s[:, self.slice_error_pos]

        robot_quat = s[:, self.slice_robot_quat]
        r, p, y = get_euler_xyz(robot_quat)
        r = check_angle(r)
        p = check_angle(p)
        y = check_angle(y)

        return error_pos[:, 2], -r, -p, -y

    def clear(self):
        self.roll_ctrl.clear()
        self.pitch_ctrl.clear()
        self.yaw_ctrl.clear()
        self.alt_ctrl.clear()


class TailsitterQRPositionControl:
    ctrl_cfg = {
        "roll": {
            "pid_param": torch.tensor([1.0, 0.01, 0.7]),
            "gain": 0.005,
            "input_clip": [-0.09, 0.09],
        },
        "pitch": {
            "pid_param": torch.tensor([1.0, 0.01, 0.7]),
            "gain": 0.005,
            "offset": 0.000225,
            "input_clip": [-0.09, 0.09],
        },
        "yaw": {
            "pid_param": torch.tensor([1.0, 0.05, 0.5]),
            "gain": 0,
            "input_clip": [-0.1, 0.1],
        },
        "x": {
            "pid_param": torch.tensor([1.0, 0.05, 0]),
            "gain": 0.005,
            "input_clip": [-0.5, 0.5],
        },
        "y": {
            "pid_param": torch.tensor([1.0, 0.05, 0]),
            "gain": 0.005,
            "input_clip": [-0.5, 0.5],
        },
        "z": {
            "pid_param": torch.tensor([1.0, 0.05, 0.5]),
            "gain": 0.007,
            "offset": 0.328,
            "input_clip": [-0.3, 0.3],
        },
    }

    def __init__(self, device):
        delta_t = 0.01

        self.roll_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["roll"],
        )
        self.pitch_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["pitch"],
        )
        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.x_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["x"],
        )
        self.y_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["y"],
        )
        self.z_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["z"],
        )

        self.slice_error_pos = slice(0, 0 + 3)
        self.slice_robot_quat = slice(3, 3 + 4)

        self.cnt = 0
        self.roll_cmd, self.pitch_cmd = None, None

        self.cmd_exist = False

    def act(self, s):
        x, y, z, roll, pitch, yaw = self.parse_observation(s)

        thrust = self.z_ctrl.action(z)

        self.cnt += 1
        if self.cnt % 4 == 0:
            x_cmd = self.x_ctrl.action(x).squeeze()
            y_cmd = self.y_ctrl.action(y).squeeze()

            self.roll_cmd = torch.sin(yaw) * x_cmd - torch.cos(yaw) * y_cmd
            self.pitch_cmd = torch.cos(yaw) * x_cmd + torch.sin(yaw) * y_cmd

            self.cnt = 0
            self.cmd_exist = True

        if not self.cmd_exist:
            self.roll_cmd = torch.zeros_like(thrust).squeeze()
            self.pitch_cmd = torch.zeros_like(thrust).squeeze()

        roll_ctrl = self.roll_ctrl.action(self.roll_cmd - roll)
        pitch_ctrl = self.pitch_ctrl.action(self.pitch_cmd - pitch)
        yaw_ctrl = self.yaw_ctrl.action(yaw)

        yaw_ctrl = torch.zeros_like(yaw_ctrl)

        return torch.clip(
            torch.concat([thrust, roll_ctrl, pitch_ctrl, yaw_ctrl], dim=1), -1, 1
        )

    def parse_observation(self, s):
        err_pos = s[:, self.slice_error_pos]
        robot_quat = s[:, self.slice_robot_quat]

        robot_ang = euler_from_quat(robot_quat)

        return (
            err_pos[..., 0],
            err_pos[..., 1],
            err_pos[..., 2],
            robot_ang[..., 0],
            robot_ang[..., 1],
            robot_ang[..., 2],
        )

    def clear(self):
        self.roll_ctrl.clear()
        self.pitch_ctrl.clear()
        self.yaw_ctrl.clear()
        self.x_ctrl.clear()
        self.y_ctrl.clear()
        self.z_ctrl.clear()

class TailsitterFWAttitudeRateControl:
    ctrl_cfg = {
        "rollrate": {
            "pid_param": torch.tensor([1.0, 0.05, 0.5]),
            "gain": 20,
            "input_clip": [-0.005, 0.005],
        },
        "pitchrate": {
            "pid_param": torch.tensor([1.0, 0.1, 0.5]),
            "gain": 100,
            "input_clip": [-0.005, 0.005],
        },
        "yawrate": {
            "pid_param": torch.tensor([1.0, 0.05, 0.5]),
            "gain": 10,
            "input_clip": [-0.005, 0.005],
        },
        "vel": {
            "pid_param": torch.tensor([1.0, 0.05, 0]),
            "gain": 0.5,
            "offset": 0.33,
        },
    }

    def __init__(self, device):
        delta_t = 0.01

        self.roll_rate_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["rollrate"],
        )
        self.pitch_rate_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["pitchrate"],
        )
        self.yaw_rate_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yawrate"],
        )
        self.vel_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

        self.slice_robot_quat = slice(3, 3 + 4)
        self.slice_robot_vel = slice(7, 7 + 3)
        self.slice_robot_angvel = slice(10, 10 + 3)
        self.slice_goal_velnorm = slice(16, 16 + 1)
        self.slice_goal_angvel = slice(17, 17 + 3)

        self.cnt = 0
        self.roll_rate_cmd, self.pitch_rate_cmd, self.yaw_rate_cmd = None, None, None

    def act(self, s):
        v, p, q, r = self.parse_observation(s)

        thrust = self.vel_ctrl.action(v)
        roll_rate_ctrl = self.roll_rate_ctrl.action(p)
        pitch_rate_ctrl = self.pitch_rate_ctrl.action(q)
        yaw_rate_ctrl = -self.yaw_rate_ctrl.action(r)

        return torch.clip(
            torch.concat(
                [thrust, roll_rate_ctrl, pitch_rate_ctrl, yaw_rate_ctrl], dim=1
            ),
            -1,
            1,
        )

    def parse_observation(self, s):
        robot_vel = s[:, self.slice_robot_vel]
        goal_velnorm = s[:, self.slice_goal_velnorm]
        vel_error = goal_velnorm - torch.norm(robot_vel, p=2, dim=-1, keepdim=True)

        robot_angvel = s[:, self.slice_robot_angvel]
        goal_angvel = s[:, self.slice_goal_angvel]
        angvel_error = goal_angvel - robot_angvel

        return (
            vel_error,
            angvel_error[..., 0],
            angvel_error[..., 1],
            angvel_error[..., 2],
        )

    def clear(self):
        self.roll_rate_ctrl.clear()
        self.pitch_rate_ctrl.clear()
        self.yaw_rate_ctrl.clear()
        self.vel_ctrl.clear()


class TailsitterFWAttitudeControl:
    ctrl_cfg = {
        "roll": {
            "pid_param": torch.tensor([1.0, 0.1, 0]),
            "gain": 31.4,
        },
        "pitch": {
            "pid_param": torch.tensor([1.0, 0.1, 0]),
            "gain": 31.4,
        },
        "yaw": {
            "pid_param": torch.tensor([1.0, 0.1, 0]),
            "gain": 31.4,
        },
        "rollrate": {
            "pid_param": torch.tensor([1.0, 0, 0.25]),
            "gain": 5,
            "input_clip": [-0.001, 0.001],
        },
        "pitchrate": {
            "pid_param": torch.tensor([1.0, 0, 0.7]),
            "gain": 150,
            "input_clip": [-0.005, 0.005],
        },
        "yawrate": {
            "pid_param": torch.tensor([1.0, 0, 0.25]),
            "gain": 1,
            "input_clip": [-0.001, 0.001],
        },
        "vel": {
            "pid_param": torch.tensor([1.0, 0.05, 0]),
            "gain": 0.5,
            "offset": 0.33,
            "input_clip": [-1, 1],
        },
    }

    def __init__(self, device):
        delta_t = 0.01

        self.roll_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["roll"],
        )
        self.pitch_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["pitch"],
        )
        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.roll_rate_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["rollrate"],
        )
        self.pitch_rate_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["pitchrate"],
        )
        self.yaw_rate_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yawrate"],
        )
        self.vel_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

        self.slice_robot_quat = slice(3, 3 + 4)
        self.slice_robot_vel = slice(7, 7 + 3)
        self.slice_robot_angvel = slice(10, 10 + 3)
        self.slice_goal_ang = slice(13, 13 + 3)
        self.slice_goal_velnorm = slice(16, 16 + 1)
        self.slice_goal_angvel = slice(17, 17 + 3)

        self.cnt = 0
        self.roll_rate_cmd, self.pitch_rate_cmd, self.yaw_rate_cmd = None, None, None

        self.cmd_exist = False

    def act(self, s):
        v, r, p, y, roll_rate, pitch_rate, yaw_rate = self.parse_observation(s)

        thrust = self.vel_ctrl.action(v)

        self.cnt += 1
        if self.cnt % 4 == 0:
            self.roll_rate_cmd = self.roll_ctrl.action(r).squeeze()
            self.pitch_rate_cmd = self.pitch_ctrl.action(p).squeeze()
            self.yaw_rate_cmd = self.yaw_ctrl.action(y).squeeze()
            self.cnt = 0
            self.cmd_exist = True

        if not self.cmd_exist:
            self.roll_rate_cmd = torch.zeros_like(thrust).squeeze()
            self.pitch_rate_cmd = torch.zeros_like(thrust).squeeze()
            self.yaw_rate_cmd = torch.zeros_like(thrust).squeeze()

        roll_rate_ctrl = self.roll_rate_ctrl.action(self.roll_rate_cmd - roll_rate)
        pitch_rate_ctrl = self.pitch_rate_ctrl.action(self.pitch_rate_cmd - pitch_rate)
        yaw_rate_ctrl = -self.yaw_rate_ctrl.action(self.yaw_rate_cmd - yaw_rate)

        return torch.clip(
            torch.concat(
                [thrust, roll_rate_ctrl, pitch_rate_ctrl, yaw_rate_ctrl], dim=1
            ),
            -1,
            1,
        )

    def parse_observation(self, s):
        vel = s[:, self.slice_robot_vel]
        goal_velnorm = s[:, self.slice_goal_velnorm]
        robot_quat = s[:, self.slice_robot_quat]
        goal_ang = s[:, self.slice_goal_ang]
        robot_angvel = s[:, self.slice_robot_angvel]

        vel_error = goal_velnorm - torch.norm(vel, p=2, dim=-1, keepdim=True)

        rb_ang = euler_from_quat(robot_quat)
        rb_ang = check_angle(rb_ang)
        goal_ang = check_angle(goal_ang)
        err_ang = goal_ang - rb_ang

        return (
            vel_error,
            err_ang[..., 0],
            err_ang[..., 1],
            err_ang[..., 2],
            robot_angvel[..., 0],
            robot_angvel[..., 1],
            robot_angvel[..., 2],
        )

    def clear(self):
        self.vel_ctrl.clear()
        self.roll_ctrl.clear()
        self.pitch_ctrl.clear()
        self.yaw_ctrl.clear()
        self.roll_rate_ctrl.clear()
        self.pitch_rate_ctrl.clear()
        self.yaw_rate_ctrl.clear()


class TailsitterFWLevelControl(TailsitterFWAttitudeControl):
    ctrl_cfg = {
        "roll": {
            "pid_param": torch.tensor([1.0, 0.1, 0]),
            "gain": 31.4,
        },
        "pitch": {
            "pid_param": torch.tensor([1.0, 0.1, 0]),
            "gain": 31.4,
        },
        "yaw": {
            "pid_param": torch.tensor([1.0, 0.1, 0]),
            "gain": 31.4,
        },
        "rollrate": {
            "pid_param": torch.tensor([1.0, 0, 0.25]),
            "gain": 5,
            "input_clip": [-0.001, 0.001],
        },
        "pitchrate": {
            "pid_param": torch.tensor([1.0, 0, 0.7]),
            "gain": 50,
            "offset": -0.13,
            "input_clip": [-0.001, 0.001],
        },
        "yawrate": {
            "pid_param": torch.tensor([1.0, 0, 0.25]),
            "gain": 1,
            "input_clip": [-0.001, 0.001],
        },
        "vel": {
            "pid_param": torch.tensor([1.0, 0.05, 0]),
            "gain": 0.033,
            "offset": 0.025,
            "input_clip": [-3, 3],
        },
    }

    def __init__(self, device):
        super().__init__(device=device)

    def act(self, s):
        v, p, y, pitch_rate, yaw_rate = self.parse_observation(s)

        thrust = self.vel_ctrl.action(v)

        self.cnt += 1
        if self.cnt % 4 == 0:
            self.pitch_rate_cmd = self.pitch_ctrl.action(p).squeeze()
            self.yaw_rate_cmd = self.yaw_ctrl.action(y).squeeze()
            self.cnt = 0
            self.cmd_exist = True

        if not self.cmd_exist:
            self.pitch_rate_cmd = torch.zeros_like(thrust).squeeze()
            self.yaw_rate_cmd = torch.zeros_like(thrust).squeeze()

        pitch_rate_ctrl = self.pitch_rate_ctrl.action(self.pitch_rate_cmd - pitch_rate)
        yaw_rate_ctrl = -self.yaw_rate_ctrl.action(self.yaw_rate_cmd - yaw_rate)

        roll_rate_ctrl = torch.zeros_like(pitch_rate_ctrl)

        return torch.clip(
            torch.concat(
                [thrust, roll_rate_ctrl, pitch_rate_ctrl, yaw_rate_ctrl], dim=1
            ),
            -1,
            1,
        )

    def parse_observation(self, s):
        robot_vel = s[:, self.slice_robot_vel]
        robot_quat = s[:, self.slice_robot_quat]
        robot_angvel = s[:, self.slice_robot_angvel]

        vel_error = 20 - torch.norm(robot_vel, p=2, dim=-1, keepdim=True)

        rb_ang = euler_from_quat(robot_quat)
        goal_ang = torch.zeros_like(rb_ang)
        goal_ang[..., 1] = 1.5
        goal_ang[..., 2] = 0
        err_ang = goal_ang - rb_ang

        return (
            vel_error,
            err_ang[..., 1],
            err_ang[..., 2],
            robot_angvel[..., 1],
            robot_angvel[..., 2],
        )


class BlimpPositionControl:
    ctrl_cfg = {
        "yaw": {
            "pid_param": torch.tensor([1.0, 0.01, 0.025]),
            "gain": 1,
        },
        "alt": {
            "pid_param": torch.tensor([1.0, 0.01, 0.5]),
            "gain": 0.2,
        },
        "vel": {
            "pid_param": torch.tensor([0.7, 0.01, 0.5]),
            "gain": 0.001,
        },
    }

    def __init__(self, device):
        delta_t = 0.1

        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

        self.slice_rb_angle = slice(0, 0 + 3)
        self.slice_goal_angle = slice(3, 3 + 3)
        self.slice_err_posNav = slice(8, 8 + 3)

    def act_on_stack(self, s_stack, k=2):  # [N, S, K]
        self.clear()
        for i in reversed(range(k)):
            a = self.act(s_stack[:, :, -i - 1])
        return a

    # def act(self, s):
    #     err_yaw, err_planar, err_z = self.parse_observation(s)

    #     yaw_ctrl = -self.yaw_ctrl.action(err_yaw)
    #     alt_ctrl = self.alt_ctrl.action(err_z)
    #     vel_ctrl = self.vel_ctrl.action(err_planar)
    #     thrust_vec = -1 * torch.ones_like(vel_ctrl)
    #     a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
    #     return a

    def act(self, s):
        err_yaw, err_planar, err_z = self.parse_observation(s)

        yaw_ctrl = -self.yaw_ctrl.action(err_yaw)
        alt_ctrl = self.alt_ctrl.action(err_z)

        vel_ctrl = torch.where(
            err_z[:, None] <= -3,
            torch.ones_like(alt_ctrl),
            self.vel_ctrl.action(err_planar),
        )
        thrust_vec = torch.where(
            err_z[:, None] <= -3,
            torch.zeros_like(vel_ctrl),
            -1 * torch.ones_like(vel_ctrl),
        )

        a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
        return a

    def parse_observation(self, s):
        error_posNav = s[:, self.slice_err_posNav]
        robot_angle = s[:, self.slice_rb_angle]

        error_navHeading = check_angle(
            compute_heading(yaw=robot_angle[:, 2], rel_pos=error_posNav)
        )
        err_planar = error_posNav[:, 0:2]
        err_planar = torch.norm(err_planar, dim=1, keepdim=True)
        err_z = error_posNav[:, 2]
        return error_navHeading, err_planar, err_z

    def clear(self):
        self.yaw_ctrl.clear()
        self.alt_ctrl.clear()
        self.vel_ctrl.clear()


class BlimpHoverControl(BlimpPositionControl):
    ctrl_cfg = {
        "yaw": {
            "pid_param": torch.tensor([1.0, 0.01, 0.025]),
            "gain": 1,
        },
        "alt": {
            "pid_param": torch.tensor([1.0, 0.01, 0.5]),
            "gain": 0.2,
        },
        "vel": {
            "pid_param": torch.tensor([0.7, 0.01, 0.5]),
            "gain": 0.005,
        },
    }

    def __init__(self, device):
        super().__init__(device)

        delta_t = 0.1

        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

        self.slice_rb_angle = slice(0, 0 + 3)
        self.slice_goal_angle = slice(3, 3 + 3)
        self.slice_err_posNav = slice(8, 8 + 3)
        self.slice_err_posHov = slice(11, 11 + 3)

    def act(self, s):
        err_yaw, err_planar, err_z = self.parse_observation(s)

        yaw_ctrl = -self.yaw_ctrl.action(err_yaw)
        alt_ctrl = self.alt_ctrl.action(err_z)

        vel_ctrl = torch.where(
            err_z[:, None] <= -3,
            torch.ones_like(alt_ctrl),
            self.vel_ctrl.action(err_planar),
        )
        thrust_vec = torch.where(
            err_z[:, None] <= -3,
            torch.zeros_like(vel_ctrl),
            -1 * torch.ones_like(vel_ctrl),
        )

        a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
        return a

    def parse_observation(self, s):
        error_posHov = s[:, self.slice_err_posHov]
        robot_angle = s[:, self.slice_rb_angle]

        error_navHeading = check_angle(
            compute_heading(yaw=robot_angle[:, 2], rel_pos=error_posHov)
        )
        err_planar = error_posHov[:, 0:2]
        err_planar = torch.norm(err_planar, dim=1, keepdim=True)
        err_z = error_posHov[:, 2]
        return error_navHeading, err_planar, err_z


class BlimpVelocityControl(BlimpPositionControl):
    ctrl_cfg = {
        "yaw": {
            "pid_param": torch.tensor([1.0, 0.01, 0.025]),
            "gain": 1,
        },
        "alt": {
            "pid_param": torch.tensor([1.0, 0.01, 0.5]),
            "gain": 0.2,
        },
        "vel": {
            "pid_param": torch.tensor([0.7, 0.01, 0.5]),
            "gain": 1.0,
        },
    }

    def __init__(self, device):
        super().__init__(device)

        delta_t = 0.1

        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

        self.slice_rb_angle = slice(0, 0 + 3)
        self.slice_goal_angle = slice(3, 3 + 3)
        self.slice_err_posNav = slice(8, 8 + 3)
        self.slice_err_posHov = slice(11, 11 + 3)
        self.slice_rb_v = slice(14, 14 + 3)
        self.slice_goal_v = slice(17, 17 + 3)

    def act(self, s):
        err_vx, err_vy, err_vz, err_z = self.parse_observation(s)

        yaw_ctrl = self.yaw_ctrl.action(err_vy)
        alt_ctrl = 5 * self.alt_ctrl.action(err_vz)

        vel_ctrl = torch.where(
            err_z[:, None] <= -3,
            torch.ones_like(alt_ctrl),
            -self.vel_ctrl.action(err_vx),
        )
        thrust_vec = torch.where(
            err_z[:, None] <= -3,
            torch.zeros_like(vel_ctrl),
            -1 * torch.ones_like(vel_ctrl),
        )
        a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)

        return a

    def parse_observation(self, s):
        rb_v = s[:, self.slice_rb_v]
        goal_v = s[:, self.slice_goal_v]
        error_posNav = s[:, self.slice_err_posNav]
        err_z = error_posNav[:, 2]

        error_v = rb_v - goal_v
        robot_angle = s[:, self.slice_rb_angle]

        err_vx, error_vy, error_vz = globalToLocalRot(
            robot_angle[:, 0],
            robot_angle[:, 1],
            robot_angle[:, 2],
            error_v[:, 0],
            error_v[:, 1],
            error_v[:, 2],
        )

        return err_vx, error_vy, error_vz, err_z


class BlimpBackwardControl(BlimpPositionControl):
    ctrl_cfg = {
        "yaw": {
            "pid_param": torch.tensor([1.0, 0.01, 0.025]),
            "gain": 1,
        },
        "alt": {
            "pid_param": torch.tensor([1.0, 0.01, 0.5]),
            "gain": 0.2,
        },
        "vel": {
            "pid_param": torch.tensor([0.7, 0.01, 0.5]),
            "gain": 0.005,
        },
    }

    def __init__(self, device):
        super().__init__(device)

        delta_t = 0.1

        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

        self.slice_rb_angle = slice(0, 0 + 3)
        self.slice_goal_angle = slice(3, 3 + 3)
        self.slice_err_posNav = slice(8, 8 + 3)
        self.slice_err_posHov = slice(11, 11 + 3)

    def act(self, s):
        err_heading, err_planar, err_z = self.parse_observation(s)

        yaw_ctrl = -self.yaw_ctrl.action(err_heading)
        alt_ctrl = -self.alt_ctrl.action(err_z)

        vel_ctrl = torch.where(
            err_z[:, None] <= -3,
            torch.ones_like(alt_ctrl),
            torch.clip(5 * self.vel_ctrl.action(err_planar) - 1, -1, -0.3),
        )
        thrust_vec = torch.where(
            err_z[:, None] <= -3,
            torch.zeros_like(vel_ctrl),
            torch.ones_like(vel_ctrl),
        )
        a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
        return a

    def parse_observation(self, s):
        error_posNav = s[:, self.slice_err_posNav]
        robot_angle = s[:, self.slice_rb_angle]

        error_navHeading = check_angle(
            compute_heading(yaw=robot_angle[:, 2], rel_pos=error_posNav) + torch.pi
        )
        err_planar = error_posNav[:, 0:2]
        err_planar = torch.norm(err_planar, dim=1, keepdim=True)
        err_z = error_posNav[:, 2]
        return error_navHeading, err_planar, err_z
