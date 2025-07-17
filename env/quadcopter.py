import math

import numpy as np
import torch

# from common.pid import QRPositionControl
from common.torch_jit_utils import *
from env.dynamics.motor import *
from env.dynamics.lift_drag import *
from env.base.goal import RandomWayPoints, FixWayPoints
from env.base.vec_env import MultiTaskVecEnv, VecEnv
from gym import spaces
from isaacgym import gymapi, gymtorch, gymutil
import copy

MASS = 1.4  # [kg]
G = 9.80665  # [m/s^2]


class Quadcopter(MultiTaskVecEnv):
    def __init__(self, cfg):
        self.max_episode_length = cfg["max_episode_length"]

        self.num_obs = 0
        self.body_id = 0  # body to observe, 0:base

        # Actions:
        # 0:4 - [thrust, roll, pitch. yaw]
        self.num_act = 2
        self.with_thrust = cfg["task"].get("with_thrust", False)
        self.with_yaw = cfg["task"].get("with_yaw", False)
        if self.with_thrust:
            self.num_act += 1
        elif self.with_yaw:
            self.num_act += 1

        # Observations: states, goals, latents
        # States: [ang, angvel] + [prev_act]
        self.num_state = 6 + self.num_act

        # Goals: [ang]
        self.num_goal = 3

        # Env Latents: [winds], [C_T, C_Q, smoothness]
        self.num_latent = 0
        self.enable_wind = cfg["dynamics"].get("enable_wind", False)
        self.num_latent += 3 if self.enable_wind else 0

        self.domain_rand = cfg["dynamics"].get("domain_rand", False)
        self.num_latent += 8 if self.domain_rand else 0

        self.num_obs = self.num_state + self.num_goal + self.num_latent

        # only useful for graph policy network
        self.initialize_graph_objects()

        self.dt = cfg["sim"]["dt"]
        self.spacing = cfg["envSpacing"]
        self.clipActions = cfg["clipActions"]

        self.clipObservations = cfg["observation"].get("clipObservations", 5.0)
        self.add_observation_noise = cfg["observation"].get("add_noise", False)
        self.scale_observation = cfg["observation"].get("scale", True)

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

        self.model_frame = cfg["dynamics"]["model_frame"]
        self.add_ground = cfg["dynamics"]["add_ground"]
        self.rotor_bodies = cfg["dynamics"]["rotor_bodies"]
        self.control_interface = cfg["dynamics"]["control_interface"]
        self.cw = torch.tensor(
            cfg["dynamics"]["cw"], device=self.sim_device, dtype=torch.float32
        )
        self.max_rotor_rad = (
            torch.tensor(
                cfg["dynamics"]["max_rotor_rpm"],
                dtype=torch.float32,
                device=self.sim_device,
            )
            * 2
            * math.pi
            / 60.0
        )
        self.control_authority = cfg["dynamics"].get("control_authority", 1)
        self.tau_up = torch.tensor(
            cfg["dynamics"]["time_constant_up"],
            dtype=torch.float32,
            device=self.sim_device,
        )
        self.tau_down = torch.tensor(
            cfg["dynamics"]["time_constant_down"],
            dtype=torch.float32,
            device=self.sim_device,
        )
        self.rotor_drag_coefficient = torch.tensor(
            cfg["dynamics"]["rotor_drag_coefficient"],
            device=self.sim_device,
            dtype=torch.float32,
        )
        self.propeller_inertia = torch.tensor(
            cfg["dynamics"]["propeller_inertia"],
            device=self.sim_device,
            dtype=torch.float32,
        )
        self.sim_to_real_ratio = torch.tensor(
            cfg["dynamics"]["sim_to_real_ratio"],
            device=self.sim_device,
            dtype=torch.float32,
        )
        self.upward = torch.tensor(
            cfg["dynamics"]["upward"], device=self.sim_device, dtype=torch.float32
        )
        self.rotor_axes = self.cw.unsqueeze(-1) * self.upward.unsqueeze(0)

        self.apply_air_drag = cfg["dynamics"].get("apply_air_drag", False)
        self.apply_rolling_moment = cfg["dynamics"].get("apply_rolling_moment", False)
        self.apply_gyroscopic_torque = cfg["dynamics"].get("apply_gyroscopic_torque", False)
        
        self.C_T = torch.tensor(
            cfg["dynamics"]["C_T"], dtype=torch.float32, device=self.sim_device
        )
        self.C_Q = torch.tensor(
            cfg["dynamics"]["C_Q"], dtype=torch.float32, device=self.sim_device
        )
        self.C_M = torch.tensor(
            cfg["dynamics"]["C_M"], dtype=torch.float32, device=self.sim_device
        )
        self.AERO_CONST = torch.tensor(
            cfg["dynamics"]["rho_A_D_pow_2"],
            dtype=torch.float32,
            device=self.sim_device,
        )

        hover_total_thrust = MASS * G
        hover_thrust_each_motor = hover_total_thrust / 4
        hover_rad_mul_sqrt_CT = torch.sqrt(hover_thrust_each_motor / self.AERO_CONST)
        self.hover_throttle_mul_sqrt_CT = hover_rad_mul_sqrt_CT / self.max_rotor_rad

        # wind
        if self.enable_wind:
            self.wind_to_velocity_ratio = torch.tensor(
                cfg["dynamics"]["wind_to_velocity_ratio"],
                device=self.sim_device,
                dtype=torch.float32,
            )
            self.body_area = torch.tensor(
                cfg["dynamics"]["body_area"],
                device=self.sim_device,
                dtype=torch.float32,
            )

        # randomize env latents
        if self.domain_rand:
            self.latent_ranges = {}

            # insert type A variable range here
            low, high = cfg["dynamics"].get("latent_range_typeA")
            self.latent_ranges["time_constant"] = [low, high]
            self.latent_ranges["C_T"] = [low, high]
            self.latent_ranges["C_Q"] = [low, high]

            # insert type B variable range here
            low, high = cfg["dynamics"].get("latent_range_typeB")
            self.latent_ranges["C_M"] = [low, high]

            # insert type C variable range here
            low, high = cfg["dynamics"].get("latent_range_typeC")
            self.latent_ranges["sim_to_real_ratio"] = [low, high]
            self.latent_ranges["thrust_imbalance"] = [low, high]
            self.latent_ranges["hover_state"] = [low, high]

            # initialize latent scale
            self.k_C_T = torch.ones(self.num_envs, 1, device=self.sim_device)
            self.k_C_Q = torch.ones(self.num_envs, 1, device=self.sim_device)
            self.k_C_M = torch.ones(self.num_envs, 1, device=self.sim_device)
            self.k_time_constant = torch.ones(self.num_envs, 1, device=self.sim_device)
            self.k_thrust_imbalance = torch.ones(
                self.num_envs, 4, device=self.sim_device
            )
            self.k_hover_state = torch.ones(self.num_envs, 1, device=self.sim_device)
            self.k_sim_to_real_ratio = torch.ones(
                self.num_envs, 1, device=self.sim_device
            )

            self.randomize_latent()
        else:
            self.k_C_T = 1
            self.k_C_Q = 1
            self.k_C_M = 1
            self.k_time_constant = 1
            self.k_hover_state = 1
            self.k_thrust_imbalance = 1
            self.k_sim_to_real_ratio = 1

        self.update_latent()

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

        # initialize tensors
        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_act),
            dtype=torch.float32,
            device=self.sim_device,
            requires_grad=False,
        )
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
        self.wind = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.sim_device,
            requires_grad=False,
        )
        self.reset()

    def initialize_graph_objects(self):
        goal_start = 6 + self.num_act
        end = goal_start + self.num_goal

        self.observation_modal_slice_and_type = [
            (slice(6, goal_start), 0),      # state: previous action
            (slice(0, 3), 1),               # state: angle
            (slice(3, 6), 2),               # state: angular velocity
            (slice(goal_start, end), 1),    # goal: angle error
        ]

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

        low, high = self.latent_ranges["C_M"]
        self.k_C_M[env_ids] = sample_from_range(
            low, high, (self.num_envs, 1), device=self.sim_device
        )

        low, high = self.latent_ranges["time_constant"]
        self.k_time_constant[env_ids] = sample_from_range(
            low, high, (self.num_envs, 1), device=self.sim_device
        )

        low, high = self.latent_ranges["thrust_imbalance"]
        self.k_thrust_imbalance[env_ids] = sample_from_range(
            low, high, (self.num_envs, 4), device=self.sim_device
        )

        low, high = self.latent_ranges["hover_state"]
        self.k_hover_state[env_ids] = sample_from_range(
            low, high, (self.num_envs, 1), device=self.sim_device
        )

        low, high = self.latent_ranges["sim_to_real_ratio"]
        self.k_sim_to_real_ratio[env_ids] = sample_from_range(
            low, high, (self.num_envs, 1), device=self.sim_device
        )

    def update_latent(self):
        self.ct = self.k_C_T * self.C_T
        self.cq = self.k_C_Q * self.C_Q
        self.cm = self.k_C_M * self.C_M
        self.time_constant_up = self.k_time_constant * self.tau_up
        self.time_constant_down = self.k_time_constant * self.tau_down
        self.sr_ratio = self.k_sim_to_real_ratio * self.sim_to_real_ratio

        self.aero_const_mul_ct = self.AERO_CONST * self.ct  # 0.004004615615878135 * 0.1
        self.hover_throttle = (
            self.k_hover_state * self.hover_throttle_mul_sqrt_CT / torch.sqrt(self.ct)
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
        asset_file = f"quadcopter/urdf/{self.model_frame}.urdf"  # f450, iris

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

            # add quadcopter here in each environment
            actor_handle = self.gym.create_actor(
                env, asset, pose, "quadcopter", i, 1, 0
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

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.goal.update_state(
            self.rb_pos[:, self.body_id], self.rb_quats[:, self.body_id]
        )

        rb_quat = self.rb_quats[env_ids, self.body_id]
        rb_euler = check_angle(euler_from_quat(rb_quat))
        rb_angvel = self.rb_angvel[env_ids, self.body_id]
        goal_ang = self.goal.ang[env_ids]

        # add noise
        if self.add_observation_noise:
            rb_euler += sampling(
                size=rb_euler.shape, scale=0.001, device=self.sim_device
            )
            rb_angvel += sampling(
                size=rb_angvel.shape, scale=0.001, device=self.sim_device
            )

        rb_angvel = global_to_local(rb_quat, rb_angvel)
        err_ang = check_angle(goal_ang - rb_euler)

        if self.scale_observation:
            rb_euler /= torch.pi
            rb_angvel /= self.avel_lim
            err_ang /= torch.pi

        self.obs_buf[env_ids, 0 : 3] = rb_euler
        self.obs_buf[env_ids, 3 : 6] = rb_angvel
        self.obs_buf[env_ids, 6 : 6 + self.num_act] = self.prev_actions[env_ids]
        n_obs = 6 + self.num_act

        # goal
        self.obs_buf[env_ids, n_obs : n_obs+3] = err_ang
        n_obs += 3

        # latents
        if self.enable_wind:
            self.obs_buf[env_ids, n_obs : n_obs + 3] = self.wind[env_ids]
            n_obs += 3

        if self.domain_rand:
            self.obs_buf[env_ids, n_obs : n_obs + 1] = self.k_C_T
            self.obs_buf[env_ids, n_obs + 1 : n_obs + 2] = self.k_C_Q
            self.obs_buf[env_ids, n_obs + 2 : n_obs + 3] = self.k_C_M
            self.obs_buf[env_ids, n_obs + 3 : n_obs + 4] = self.k_time_constant

            ll = self.k_thrust_imbalance[env_ids, 0:1]
            lr = self.k_thrust_imbalance[env_ids, 1:2]
            tl = self.k_thrust_imbalance[env_ids, 2:3]
            tr = self.k_thrust_imbalance[env_ids, 3:4]
            self.obs_buf[env_ids, n_obs + 4 : n_obs + 5] = (
                tl + tr - ll + lr
            )  # thrust imbalance in roll
            self.obs_buf[env_ids, n_obs + 5 : n_obs + 6] = (
                ll + tl - lr - tr
            )  # thrust imbalance in pitch

            self.obs_buf[env_ids, n_obs + 6 : n_obs + 7] = self.k_hover_state
            self.obs_buf[env_ids, n_obs + 7 : n_obs + 8] = self.k_sim_to_real_ratio

            n_obs += 8

        return torch.clip(self.obs_buf, -self.clipObservations, self.clipObservations)

    def get_reward(self):
        (
            self.reward_buf,
            self.reset_buf,
            self.return_buf,
        ) = compute_quadcopter_reward(
            root_positions=self.rb_pos[:, self.body_id],
            root_quats=self.rb_quats[:, self.body_id],
            root_linvels=self.rb_vel[:, self.body_id],
            root_angvels=self.rb_angvel[:, self.body_id],
            goal_positions=self.goal.pos,
            pos_lim=self.pos_lim,
            vel_lim=self.vel_lim,
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

        self.goal.sample(self.difficulty)

        if self.domain_rand:
            self.randomize_latent()
            self.update_latent()

        if self.enable_wind:
            self.wind = sample_from_range(
                -1, 1, (self.num_envs, 3), device=self.sim_device
            )

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
                difficulty=self.difficulty,
                max_difficulty=self.max_difficulty,
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

        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_act),
            dtype=torch.float32,
            device=self.sim_device,
            requires_grad=False,
        )

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.rb_states[:: self.num_bodies].contiguous()),
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
        #####

        actions = torch.clamp(actions, -self.clipActions, self.clipActions)
        self.prev_actions = actions

        actions = self.control_authority * self.sr_ratio * actions

        if not self.with_thrust:
            actions = F.pad(input=actions, pad=(1, 0), mode="constant", value=0)
        if not self.with_yaw:
            actions = F.pad(input=actions, pad=(0, 1), mode="constant", value=0)

        if self.control_interface == "rotors":
            self.actuators = first_order_system(
                actions,
                self.actuators,
                self.dt,
                self.time_constant_up,
                self.time_constant_down,
            )
        elif self.control_interface == "attitude":
            self.actuators = first_order_system(
                attitude_control(actions),
                self.actuators,
                self.dt,
                self.time_constant_up,
                self.time_constant_down,
            )

        virtual_rotor = self.hover_throttle + self.actuators
        virtual_rotor = torch.clamp(virtual_rotor, 0, 1)
        rotor_rad = self.max_rotor_rad * virtual_rotor

        self.forces[:] = 0.0  # body frame
        self.torques[:] = 0.0  # body frame

        if self.enable_wind:
            self.wind = 0.99 * self.wind + 0.01 * sample_from_range(
                -1, 1, (self.num_envs, 3), device=self.sim_device
            )
            wind_velocity = self.wind_to_velocity_ratio * self.wind
            wind_force = simulate_wind(
                self.rb_quats[:, self.body_id], wind_velocity, self.body_area
            )
            self.forces[:, self.body_id] += wind_force
        else:
            wind_velocity = torch.zeros_like(self.forces[:, self.body_id])

        thrusts, torques, air_drag, rolling_moment, gyroscopic_torque = simulate_rotors(
            rotor_rad,
            self.rb_quats[:, self.body_id],
            self.rb_angvel[:, self.body_id],
            self.rb_vel[:, self.body_id],
            wind_velocity,
            self.cw,
            self.aero_const_mul_ct,
            self.cq,
            self.cm,
            self.rotor_drag_coefficient,
            self.propeller_inertia,
            self.upward,
            self.rotor_axes,
            enable_air_drag=self.apply_air_drag, # return None if False
            enable_rolling_moment=self.apply_rolling_moment, # return None if False
            enable_gyroscopic_torque=self.apply_gyroscopic_torque, # return None if False
        )

        thrusts = self.k_thrust_imbalance * thrusts

        self.forces[:, self.rotor_bodies, 2] += thrusts
        self.torques[:, self.rotor_bodies, 2] += torques
        if self.apply_air_drag:
            self.forces[:, self.rotor_bodies] += air_drag
        if self.apply_rolling_moment:
            self.torques[:, self.rotor_bodies] += rolling_moment
        if self.apply_gyroscopic_torque:
            self.torques[:, self.rotor_bodies] += gyroscopic_torque

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

        if self.visualize_force:  # thrusts
            for i in self.rotor_bodies:
                num_lines, line_colors, line_vertices = self._add_force_lines(
                    num_lines, line_colors, line_vertices, i
                )

        return line_vertices, line_colors, num_lines

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


############################################################


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_quadcopter_reward(
    root_positions,
    root_quats,
    root_linvels,
    root_angvels,
    goal_positions,
    pos_lim,
    vel_lim,
    reset_buf,
    progress_buf,
    max_episode_length,
    return_buf,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor]

    # distance to target
    target_dist = torch.norm(
        root_positions[..., :2] - goal_positions[..., :2], p=2, dim=-1
    )
    vel_norm = torch.norm(root_linvels, p=2, dim=-1)
    # pos_reward = 1.0 / (1.0 + target_dist * target_dist)

    # # uprightness
    # ups = quat_axis(root_quats, 2)
    # tiltage = torch.abs(1 - ups[..., 2])
    # up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # # spinning
    # spinnage = torch.abs(root_angvels[..., 2])
    # spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # combined reward
    # uprigness and spinning only matter when close to the target
    # reward = pos_reward + pos_reward * (up_reward + spinnage_reward)
    reward = torch.zeros(1, device="cuda")

    # return_buf += reward

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > pos_lim, ones, die)
    # die = torch.where(root_positions[..., 2] < 0.3, ones, die)
    die = torch.where(vel_norm > vel_lim, ones, die)
    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset, return_buf


@torch.jit.script
def attitude_control(actions):
    # type: (Tensor) -> Tensor
    # actions: [thrust, roll, pitch, yaw]
    # rotors: [BL-, BR+, FL+, FR-]

    thrust = actions[:, 0]
    roll = actions[:, 1]
    pitch = actions[:, 2]
    yaw = actions[:, 3]

    rotor1 = thrust + roll + pitch - yaw
    rotor2 = thrust - roll + pitch + yaw
    rotor3 = thrust + roll - pitch + yaw
    rotor4 = thrust - roll - pitch - yaw
    return torch.stack([rotor1, rotor2, rotor3, rotor4], dim=-1)
