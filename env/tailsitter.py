import math

import numpy as np
import torch

from common.torch_jit_utils import *
from env.base.goal import RandomWayPoints, FixWayPoints
from env.base.vec_env import MultiTaskVecEnv, VecEnv
from gym import spaces
from isaacgym import gymapi, gymtorch, gymutil


class Tailsitter(MultiTaskVecEnv):
    def __init__(self, cfg):
        self.max_episode_length = cfg["max_episode_length"]

        self.num_obs = 0
        self.body_id = 0  # body to observe, 0:base

        # States: quat, angvel, vel, err_ang, err_vnorm
        self.num_state = 10

        # Goals: none
        self.num_goal = 0

        # Env Latents: C_T, C_Q, smoothness
        self.domain_rand = cfg["dynamics"].get("domain_rand", False)
        self.num_latent = 3 if self.domain_rand else 0

        self.num_obs = self.num_state + self.num_goal + self.num_latent

        # Actions:
        # 0:4 - [thrust, roll, pitch. yaw]
        self.num_act = 4

        self.dt = cfg["sim"]["dt"]
        self.spacing = cfg["envSpacing"]
        self.clipObservations = cfg["clipObservations"]
        self.clipActions = cfg["clipActions"]

        self.spawn_height = cfg["task"].get("spawn_height", 2)
        self.pos_lim = cfg["task"].get("pos_lim", 4)
        self.max_height = cfg["task"].get("max_height", 4)
        self.hover_zone = cfg["task"].get("hover_zone", 10)
        self.vel_lim = cfg["task"].get("vel_lim", 3)
        self.avel_lim = cfg["task"].get("avel_lim", 4 * math.pi)

        self.init_pos = cfg["task"].get("init_pos", [0, 0, 0])
        self.init_pos[2] += self.spawn_height
        self.init_vel = cfg["task"].get("init_vel", [0, 0, 0])
        self.init_ang = cfg["task"].get("init_ang", [0, 0, 0])
        self.init_avel = cfg["task"].get("init_avel", [0, 0, 0])

        self.rand_rb_pos = cfg["task"].get("rand_rb_pos", False)
        self.rand_rb_vel = cfg["task"].get("rand_rb_vel", False)
        self.rand_rb_ang = cfg["task"].get("rand_rb_ang", False)
        self.rand_rb_avel = cfg["task"].get("rand_rb_avel", False)
        self.rand_rb_actuator = cfg["task"].get("rand_rb_actuator", False)

        self.target_velnorm = cfg["goal"].get("target_velnorm", 20)

        super().__init__(cfg=cfg)
        self.obs_space = spaces.Box(
            np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf
        )
        self.act_space = spaces.Box(
            np.ones(self.num_act) * -1.0, np.ones(self.num_act) * 1.0
        )

        # parameters
        self.visualize_goal = cfg["goal"]["visualize_goal"]
        self.visualize_force = cfg["dynamics"]["visualize_force"]
        self.visualize_aeroforce = cfg["dynamics"]["visualize_aeroforce"]

        self.add_ground = cfg["dynamics"]["add_ground"]
        self.rotor_bodies = cfg["dynamics"]["rotor_bodies"]
        self.control_interface = cfg["dynamics"]["control_interface"]
        self.hover_actuator = torch.tensor(
            [cfg["dynamics"]["hover_actuator"] for _ in range(4)],
            device=self.sim_device,
            dtype=torch.float32,
        )
        self.cw = torch.tensor(
            cfg["dynamics"]["cw"], device=self.sim_device, dtype=torch.float32
        )
        self.actuator_to_avel_ratio = cfg["dynamics"]["actuator_to_avel_ratio"]

        self.drag_bodies = torch.tensor(
            cfg["dynamics"]["drag_bodies"], device=self.sim_device, dtype=torch.long
        )
        self.area = torch.tensor(
            cfg["dynamics"]["area"], device=self.sim_device, dtype=torch.float32
        )
        self.alpha0 = torch.tensor(
            cfg["dynamics"]["alpha0"], device=self.sim_device, dtype=torch.float32
        )
        self.cla = torch.tensor(
            cfg["dynamics"]["cla"], device=self.sim_device, dtype=torch.float32
        )
        self.cda = torch.tensor(
            cfg["dynamics"]["cda"], device=self.sim_device, dtype=torch.float32
        )
        self.alphaStall = torch.tensor(
            cfg["dynamics"]["alphaStall"], device=self.sim_device, dtype=torch.float32
        )
        self.claStall = torch.tensor(
            cfg["dynamics"]["claStall"], device=self.sim_device, dtype=torch.float32
        )
        self.cdaStall = torch.tensor(
            cfg["dynamics"]["cdaStall"], device=self.sim_device, dtype=torch.float32
        )
        self.forward = torch.tensor(
            cfg["dynamics"]["forward"], device=self.sim_device, dtype=torch.float32
        )
        self.upward = torch.tensor(
            cfg["dynamics"]["upward"], device=self.sim_device, dtype=torch.float32
        )
        self.rho = torch.tensor(
            cfg["dynamics"]["rho"], device=self.sim_device, dtype=torch.float32
        )

        # k_* are randomizable latents
        self.C_T = torch.tensor(
            cfg["dynamics"]["C_T"], dtype=torch.float32, device=self.sim_device
        )
        self.C_Q = torch.tensor(
            cfg["dynamics"]["C_Q"], dtype=torch.float32, device=self.sim_device
        )
        self.smooth = self.dt * torch.tensor(
            cfg["dynamics"]["smooth"], device=self.sim_device
        )

        # randomize env latents
        if self.domain_rand:
            low, high = cfg["dynamics"].get("latent_range_typeA", [0.8, 1.25])
            self.latent_ranges = {}
            self.latent_ranges["C_T"] = [low, high]
            self.latent_ranges["C_Q"] = [low, high]

            low, high = cfg["dynamics"].get("latent_range_typeB", [0.5, 2.0])
            self.latent_ranges["smooth"] = [low, high]

            self.k_C_T = self.C_T.repeat(self.num_envs, 1)
            self.k_C_Q = self.C_Q.repeat(self.num_envs, 1)
            self.k_smooth = self.smooth.repeat(self.num_envs, 1)

            self.randomize_latent()

        # initialize goal and task
        if cfg["goal"]["goal_type"] == "fix":
            self.goal = FixWayPoints(
                device=self.sim_device,
                num_envs=self.num_envs,
                hov_height=cfg["goal"].get("hov_height", self.spawn_height),
                nav_scale=self.pos_lim
                * 0.5,  # waypoints positions to the center, only matter if position_target_type is waypoints
                **cfg["goal"],
            )
        else:
            self.goal = RandomWayPoints(
                device=self.sim_device,
                num_envs=self.num_envs,
                hov_height=cfg["goal"].get("hov_height", self.spawn_height),
                nav_scale=self.pos_lim * 0.5,
                **cfg["goal"],
            )

        self.set_task(task=self._task_mode)

        # initialise envs and state tensors
        self.create_envs()

        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.prepare_sim(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.rb_states = gymtorch.wrap_tensor(rb_state_tensor)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.rb_pos = self.rb_states[:, 0:3].view(self.num_envs, self.num_bodies, 3)
        self.rb_quats = self.rb_states[:, 3:7].view(self.num_envs, self.num_bodies, 4)
        self.rb_vel = self.rb_states[:, 7:10].view(self.num_envs, self.num_bodies, 3)
        self.rb_angvel = self.rb_states[:, 10:13].view(
            self.num_envs, self.num_bodies, 3
        )

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]

        self.actuators = torch.zeros(
            (self.num_envs, 4),
            dtype=torch.float32,
            device=self.sim_device,
            requires_grad=False,
        )
        self.forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            dtype=torch.float32,
            device=self.sim_device,
            requires_grad=False,
        )
        self.torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            dtype=torch.float32,
            device=self.sim_device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros(
            (self.num_envs, 4),
            dtype=torch.float32,
            device=self.sim_device,
            requires_grad=False,
        )
        self.wind = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.sim_device,
            requires_grad=False,
        )

        # self.controllers = []
        # self.controllers.append(TailsitterQRPositionControl(device=self.sim_device))
        # self.controllers.append(TailsitterFWLevelControl(device=self.sim_device))

        self.reset()

    def randomize_latent(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        low, high = self.latent_ranges["C_T"]
        self.k_C_T[env_ids] = sample_from_range(
            low, high, (self.num_envs, 1), device=self.sim_device
        )

        low, high = self.latent_ranges["C_Q"]
        self.k_C_Q[env_ids] = sample_from_range(
            low, high, (self.num_envs, 1), device=self.sim_device
        )

        low, high = self.latent_ranges["smooth"]
        self.k_smooth[env_ids] = sample_from_range(
            low, high, (self.num_envs, 1), device=self.sim_device
        )

    def create_envs(self):
        # add ground plane
        if self.add_ground:
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            plane_params.restitution = 1
            plane_params.static_friction = 0
            plane_params.dynamic_friction = 0
            self.gym.add_ground(self.sim, plane_params)

        # define environment space (for visualisation)
        lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0)
        upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        # add assets
        asset_root = "assets"
        asset_file = "tailsitter/urdf/model.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = True
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = self.avel_lim
        asset_options.slices_per_cylinder = 40
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)

        # define asset pose
        pose = gymapi.Transform()
        pose.p.z = self.spawn_height  # generate the blimp h m from the ground
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # generate environments
        self.actor_handles = []
        self.envs = []
        print(f"Creating {self.num_envs} environments.")
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # add tailsitter here in each environment
            actor_handle = self.gym.create_actor(
                env, asset, pose, "tailsitter", i, 1, 0
            )

            dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"].fill(1000.0)
            dof_props["damping"].fill(0.0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)

            self.envs.append(env)
            self.actor_handles.append(actor_handle)

    def get_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.goal.update_state(
            self.rb_pos[:, self.body_id], self.rb_quats[:, self.body_id]
        )

        rb_quat = self.rb_quats[env_ids, self.body_id]
        # rb_vel_local = global_to_local(rb_quat, self.rb_vel[env_ids, self.body_id])
        rb_vel_global = self.rb_vel[env_ids, self.body_id]
        rb_angvel_local = global_to_local(
            rb_quat, self.rb_angvel[env_ids, self.body_id]
        )
        # err_angle = check_angle(self.goal.ang[env_ids] - euler_from_quat(rb_quat))
        # err_vnorm = self.goal.velnorm[env_ids] - rb_vel_local[..., 2:3]

        self.obs_buf[env_ids, 0:4] = rb_quat
        self.obs_buf[env_ids, 4:7] = rb_angvel_local/self.avel_lim
        self.obs_buf[env_ids, 7:10] = rb_vel_global/self.vel_lim  # local

        # self.obs_buf[env_ids, 10:13] = err_angle/torch.pi
        # self.obs_buf[env_ids, 13:14] = err_vnorm/self.vel_lim

        # self.obs_buf[env_ids, 14:18] = self.prev_actions[env_ids]

        # env state
        # self.obs_buf[env_ids, 18:22] = self.actuators[env_ids]

        # d = self.idx_expr
        # for controller in self.controllers:
        #     a = controller.act(self.obs_buf)
        #     self.obs_buf[env_ids, d : d + self.num_act] = a[env_ids]
        #     d += self.num_act

        # latents
        if self.domain_rand:
            self.obs_buf[env_ids, 10:11] = self.k_C_T
            self.obs_buf[env_ids, 11:12] = self.k_C_Q
            self.obs_buf[env_ids, 12:13] = self.k_smooth

        return torch.clip(self.obs_buf, -self.clipObservations, self.clipObservations)

    def get_reward(self):
        (
            self.reward_buf,
            self.reset_buf,
            self.return_buf,
        ) = compute_reward(
            root_positions=self.rb_pos[:, self.body_id],
            root_quats=self.rb_quats[:, self.body_id],
            root_linvels=self.rb_vel[:, self.body_id],
            root_angvels=self.rb_angvel[:, self.body_id],
            goal_positions=self.goal.pos,
            goal_velocities=self.goal.vel,
            goal_angles=self.goal.ang,
            goal_angvel=self.goal.angvel,
            actuators=self.actuators,
            pos_lim=self.pos_lim,
            max_height=self.max_height,
            vel_lim=self.vel_lim,
            avel_lim=self.avel_lim,
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            max_episode_length=self.max_episode_length,
            return_buf=self.return_buf,
        )

        self.reset_buf = self.task.reset_tasks(
            self.reset_buf,
            task_mode=self._task_mode,
            rb_pos=self.rb_pos[:, self.body_id],
            rb_vel=self.rb_vel[:, self.body_id],
            rb_quats=self.rb_quats[:, self.body_id],
            rb_angvel=self.rb_angvel[:, self.body_id],
            goal_pos=self.goal.pos,
            goal_vel=self.goal.vel,
            goal_velnorm=self.goal.velnorm,
            goal_ang=self.goal.ang,
            goal_angvel=self.goal.angvel,
        )

    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) == 0:
            return

        self.set_difficulty()
        # [controller.clear() for controller in self.controllers]

        self.goal.sample(self.difficulty)

        if self.domain_rand:
            self.randomize_latent()

        if (
            self.rand_rb_pos
            or self.rand_rb_vel
            or self.rand_rb_ang
            or self.rand_rb_avel
        ):
            (
                pos,
                vel,
                ang,
                angvel,
                self.goal.pos[env_ids],
                self.goal.vel[env_ids],
                self.goal.velnorm[env_ids],
                self.goal.ang[env_ids],
                self.goal.angvel[env_ids],
            ) = self.task.sample_robot_state(
                pos_lim=self.pos_lim,
                vel_lim=self.vel_lim,
                avel_lim=self.avel_lim,
                goal_pos=self.goal.pos[env_ids],
                goal_vel=self.goal.vel[env_ids],
                goal_velnorm=self.goal.velnorm[env_ids],
                goal_ang=self.goal.ang[env_ids],
                goal_angvel=self.goal.angvel[env_ids],
                num_bodies=self.num_bodies,
                mode=self._task_mode,
                env_ids=env_ids,
                size=len(env_ids),
                spawn_height=self.spawn_height,
                max_height=self.max_height,
                hover_zone=self.hover_zone,
                target_velnorm=self.target_velnorm,
                difficulty=self.difficulty,
                max_difficulty=self.max_difficulty,
                forward=self.forward[0],
            )
        if not self.rand_rb_pos:
            pos = torch.tile(
                torch.tensor(
                    self.init_pos, device=self.sim_device, dtype=torch.float32
                ),
                (len(env_ids), self.num_bodies, 1),
            )
        if not self.rand_rb_vel:
            vel = torch.tile(
                torch.tensor(
                    self.init_vel, device=self.sim_device, dtype=torch.float32
                ),
                (len(env_ids), self.num_bodies, 1),
            )
        if not self.rand_rb_ang:
            ang = torch.tile(
                torch.tensor(
                    self.init_ang, device=self.sim_device, dtype=torch.float32
                ),
                (len(env_ids), self.num_bodies, 1),
            )
        if not self.rand_rb_avel:
            angvel = torch.tile(
                torch.tensor(
                    self.init_avel, device=self.sim_device, dtype=torch.float32
                ),
                (len(env_ids), self.num_bodies, 1),
            )

        quat = quat_from_euler(ang)

        self.rb_pos[env_ids] = pos
        self.rb_quats[env_ids] = quat
        self.rb_vel[env_ids] = vel
        self.rb_angvel[env_ids] = angvel

        # clear actions for reset envs
        if self.rand_rb_actuator:
            self.actuators[env_ids] = sampling((len(env_ids), 4), 0.05, self.sim_device)
        else:
            self.actuators[env_ids] = 0.0
        self.forces[env_ids] = 0.0
        self.torques[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.rb_states[::self.num_bodies].contiguous()),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.return_buf[env_ids] = 0
        self.truncated_buf[env_ids] = 0

        self.get_obs()

    def step(self, actions):
        actions = actions.to(self.sim_device)

        ##### testing
        # actions = torch.zeros_like(actions)
        # ctrl = self.controllers[0].act(self.get_obs())
        # actions = ctrl
        #####

        actions = torch.clamp(actions, -self.clipActions, self.clipActions)
        self.prev_actions = actions

        smooth = self.k_smooth*self.smooth
        if self.control_interface == "rotors":
            self.actuators = (1 - smooth) * self.actuators + smooth * actions
        elif self.control_interface == "attitude":
            self.actuators = (
                1 - smooth
            ) * self.actuators + smooth * attitude_control(actions)

        actuators = self.hover_actuator + self.actuators

        actuators = torch.clamp(actuators, 0, 1)
        rotor_avel = self.actuator_to_avel_ratio * actuators

        self.forces[:] = 0.0  # body frame
        self.torques[:] = 0.0  # body frame

        rotor_thrusts, rotor_torques = simulate_rotors(
            rotor_avel, self.k_C_T*self.C_T, self.k_C_Q*self.C_Q, self.cw
        )
        self.forces[:, self.rotor_bodies, 2] += rotor_thrusts
        self.torques[:, self.rotor_bodies, 2] += rotor_torques

        # randomize wind
        # self.wind = torch.normal(mean=self.k_wind_mean, std=self.k_wind_std)

        liftdrag_force = simulate_aerodynamics(
            rb_quats=self.rb_quats[:, self.drag_bodies],
            rb_vel=self.rb_vel[:, self.drag_bodies],
            wind=self.wind,
            alpha0=self.alpha0,
            cla=self.cla,
            cda=self.cda,
            alphaStall=self.alphaStall,
            claStall=self.claStall,
            cdaStall=self.cdaStall,
            forward=self.forward,
            upward=self.upward,
            area=self.area,
            rho=self.rho,
        )
        self.forces[:, self.drag_bodies] += liftdrag_force

        # apply actions
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.forces),
            gymtorch.unwrap_tensor(self.torques),
            gymapi.LOCAL_SPACE,
        )

        # simulate and render
        self.simulate()
        if not self.headless:
            self.render()

        # reset environments if required
        self.progress_buf += 1

        self.get_obs()
        self.get_reward()

    def _generate_lines(self):
        num_lines = 0
        line_colors = []
        line_vertices = []

        if self.visualize_goal:  # position goal
            num_lines, line_colors, line_vertices = self._add_goal_lines(
                num_lines, line_colors, line_vertices
            )

        if self.visualize_aeroforce:  # aerodynamics, body 0,5,6,7,8
            num_lines, line_colors, line_vertices = self._add_force_lines(
                num_lines, line_colors, line_vertices, 0, s=100, color=[0, 255, 0]
            )

        if self.visualize_force:  # thrusts
            for i in self.rotor_bodies:
                num_lines, line_colors, line_vertices = self._add_force_lines(
                    num_lines, line_colors, line_vertices, i
                )

        return line_vertices, line_colors, num_lines

    def _add_goal_lines(self, num_lines, line_colors, line_vertices, color=[0, 0, 255]):
        num_lines += 1
        line_colors += [color]

        for i in range(self.num_envs):
            vertices = [
                [
                    self.goal.pos[i, 0].item(),
                    self.goal.pos[i, 1].item(),
                    0,
                ],
                [
                    self.goal.pos[i, 0].item(),
                    self.goal.pos[i, 1].item(),
                    self.goal.pos[i, 2].item(),
                ],
            ]
            if len(line_vertices) > i:
                line_vertices[i] += vertices
            else:
                line_vertices.append(vertices)

        return num_lines, line_colors, line_vertices

    def _add_force_lines(
        self, num_lines, line_colors, line_vertices, idx, s=1, color=[255, 0, 0]
    ):
        num_lines += 1
        line_colors += [color]

        forces = local_to_global(self.rb_quats[:, idx], self.forces[:, idx])

        for i in range(self.num_envs):
            vertices = [
                [
                    self.rb_pos[i, idx, 0].item(),
                    self.rb_pos[i, idx, 1].item(),
                    self.rb_pos[i, idx, 2].item(),
                ],
                [
                    self.rb_pos[i, idx, 0].item() + s * forces[i, 0].item(),
                    self.rb_pos[i, idx, 1].item() + s * forces[i, 1].item(),
                    self.rb_pos[i, idx, 2].item() + s * forces[i, 2].item(),
                ],
            ]
            if len(line_vertices) > i:
                line_vertices[i] += vertices
            else:
                line_vertices.append(vertices)

        return num_lines, line_colors, line_vertices


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_reward(
    root_positions,
    root_quats,
    root_linvels,  # global frame
    root_angvels,
    goal_positions,
    goal_velocities,  # global frame
    goal_angles,
    goal_angvel,
    actuators,
    pos_lim,
    max_height,
    vel_lim,
    avel_lim,
    reset_buf,
    progress_buf,
    max_episode_length,
    return_buf,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor]

    # r, p, y = get_euler_xyz(root_quats)

    # # map roll to +-180 deg
    # r = torch.where(r > torch.pi, r - 2*torch.pi, r)
    # r = torch.where(r < -torch.pi, r + 2*torch.pi, r)

    # # map pitch to +-180 deg
    # p = torch.where(p > torch.pi, p - 2*torch.pi, p)
    # p = torch.where(p < -torch.pi, p + 2*torch.pi, p)

    # # map yaw to +-180 deg
    # y = torch.where(y > torch.pi, y - 2*torch.pi, y)
    # y = torch.where(y < -torch.pi, y + 2*torch.pi, y)

    # distance to target
    target_dist = torch.norm(
        root_positions[..., :2] - goal_positions[..., :2], p=2, dim=-1
    )

    # target_dist = torch.norm(root_positions[...,:2] - goal_positions[...,:2], p=2, dim=-1)
    # planar_reward = 1.0 / (1.0 + target_dist**2)

    # z_dist = torch.abs(root_positions[...,2] - goal_positions[...,2])
    # z_reward = 1.0 / (1.0 + z_dist**2)

    # ang
    # r_err = r - goal_angles[...,0]
    # p_err = p - goal_angles[...,1]
    # y_err = y - goal_angles[...,2]
    # r_reward = 1.0 / (1.0 + r_err**2)
    # p_reward = 1.0 / (1.0 + p_err**2)
    # y_reward = 1.0 / (1.0 + y_err**2)
    # ang_reward = 1.0 / (1.0 + p_err**2)

    # uprightness
    # ups = quat_axis(root_quats, 2)
    # tiltage = torch.abs(1 - ups[..., 2])
    # up_reward = 1.0 / (1.0 + tiltage**2)

    # spinning
    # spinnage = torch.norm(root_angvels, p=2, dim=-1)
    # spinnage_reward = 1.0 / (1.0 + spinnage**2)

    # velocity
    # velocity_reward = torch.norm(root_linvels-goal_velocities, p=2, dim=-1)
    # velocity_reward = 1.0 / (1.0 + velocity_reward**2)

    # travel_reward = root_positions[...,0]
    # speed_reward = root_linvels[...,0]

    # combined reward
    # reward = pos_reward + pos_reward * (up_reward + spinnage_reward)
    # reward = 0.1*speed_reward + p_reward*velocity_reward + p_reward*velocity_i_reward + p_reward + spinnage_reward
    # reward = 0.1*travel_reward + 0.1*speed_reward + p_reward*velocity_reward + p_reward*velocity_i_reward + r_reward + p_reward + spinnage_reward
    # print("aaa", travel_reward[0], speed_reward[0], p_reward[0], spinnage_reward[0])
    # print("bbb", root_linvels[0], goal_velocities[0])
    reward = torch.zeros(1, device="cuda")

    # return_buf += reward

    # resets due to misbehavior
    vel_norm = torch.norm(root_linvels, p=2, dim=-1)
    avel_norm = torch.norm(root_angvels, p=2, dim=-1)

    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    # die = torch.where(root_positions[..., 2] < 0.5, ones, die)
    die = torch.where(root_positions[..., 2] > max_height, ones, die)
    die = torch.where(root_positions[..., 2] < -max_height, ones, die)
    die = torch.where(target_dist > pos_lim, ones, die)
    die = torch.where(vel_norm > vel_lim, ones, die)
    die = torch.where(avel_norm > avel_lim, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset, return_buf

@torch.jit.script
def attitude_control(actions):
    # type: (Tensor) -> Tensor
    # actions: [thrust, roll, pitch, yaw]
    # rotors: [top right-, low left-, top left+, low right+]

    thrust = actions[:, 0]
    roll = actions[:, 1]
    pitch = actions[:, 2]
    yaw = actions[:, 3]

    rotor1 = thrust - roll + pitch - yaw
    rotor2 = thrust + roll - pitch - yaw
    rotor3 = thrust + roll + pitch + yaw
    rotor4 = thrust - roll - pitch + yaw

    return torch.stack([rotor1, rotor2, rotor3, rotor4], dim=-1)
