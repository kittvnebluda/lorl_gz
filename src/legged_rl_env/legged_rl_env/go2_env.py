from enum import Enum, auto
from math import inf
from time import perf_counter, sleep

import gymnasium as gym
import numpy as np
import rclpy.executors
from geometry_msgs.msg import Pose
from gymnasium import spaces
from numpy import deg2rad, exp, float32, pi
from rclpy.duration import Duration
from rclpy.time import Time

from legged_rl_env.constants import (
    jpos_high,
    jpos_low,
    jvel_max,
    stable_standing_joint_positions,
)
from legged_rl_env.go2_node import Go2Node
from legged_rl_env.utils import (
    ServiceCall,
    bool_res_handler,
    denorm11,
    norm01,
    norm11,
    quat_msg_from_euler,
    wait_for_all_futures,
)


class Done(Enum):
    ALIVE = auto()
    TERMINATED = auto()
    TRUNCATED = auto()


class Go2Env(gym.Env):
    metadata = {}

    reset_srv_timeout = 10.0

    # Default joint positions
    # TODO: Calculate them for target body height
    q_homing = np.array(stable_standing_joint_positions, dtype=float32)
    q_homing_normalized = norm11(q_homing, jpos_low, jpos_high).astype(float32)

    def __init__(
        self,
        step_delay: float = 0.02,
        reset_delay: float = 0.1,
        real_time_factor: float = 1,
        learning_starts: int = 0,
        wait_for_services=True,
        learning=True,
        render_mode=None,
    ) -> None:
        self.step_delay = step_delay
        self.reset_delay = reset_delay
        self.rtf = real_time_factor
        self.learning_starts = learning_starts if learning else inf
        self.render_mode = render_mode

        self.ref_vx_max = 0.50
        self.ref_vy_max = 0.25
        self.ref_wz_max = 2.00
        self.ref_vel_max = np.array(
            [self.ref_vx_max, self.ref_vy_max, self.ref_wz_max],
            dtype=float32,
        )
        self.lin_vel_low = np.array([-3.5, -2.0, -1.0], dtype=float32)
        self.lin_vel_high = np.array([3.7, 2.0, 1.0], dtype=float32)
        self.ang_vel_max = np.array([4.0, 4.0, 3.0], dtype=float32)

        self.ref_z_low = 0.0
        self.ref_z_high = 0.5

        self.observation_space = spaces.Dict(
            {
                "linear_vel": spaces.Box(low=-1, high=1, shape=(3,), dtype=float32),
                "angular_vel": spaces.Box(low=-1, high=1, shape=(3,), dtype=float32),
                "orientation": spaces.Box(low=-1, high=1, shape=(2,), dtype=float32),
                "joint_positions": spaces.Box(
                    low=-1, high=1, shape=(12,), dtype=float32
                ),
                "joint_velocities": spaces.Box(
                    low=-1, high=1, shape=(12,), dtype=float32
                ),
                "ref_vel": spaces.Box(low=-1, high=1, shape=(3,), dtype=float32),
                "ref_z": spaces.Box(low=0, high=1, shape=(1,), dtype=float32),
                "prev_action": spaces.Box(low=-1, high=1, shape=(12,), dtype=float32),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=float32)

        # Termination and truncation thresholds
        self.roll_th = deg2rad(36)
        self.pitch_th = deg2rad(36)
        self.z_min_th = 0.20
        self.max_ep_time = 10

        # ROS
        self.node = Go2Node(wait_for_services=wait_for_services)
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)

        # Grid Adaptive Curriculum
        self.vx_grid = np.arange(-self.ref_vx_max, self.ref_vx_max + 0.1, 0.1)
        self.wz_grid = np.arange(-self.ref_wz_max, self.ref_wz_max + 0.1, 0.1)
        if learning:
            self.active_mask = np.zeros(
                (len(self.vx_grid), len(self.wz_grid)), dtype=bool
            )
            cx = len(self.vx_grid) // 2
            cw = len(self.wz_grid) // 2
            self.active_mask[cx - 1 : cx + 2, cw - 1 : cw + 2] = True
        else:
            self.active_mask = np.ones(
                (len(self.vx_grid), len(self.wz_grid)), dtype=bool
            )

        self.tb_log = {}  # Logs for tensorboard

        self._prev_action = np.zeros((12,), dtype=float32)
        self._ref_vel = np.array([0.0, 0.0, 0.0], dtype=float32)  # vx,vy,wz
        self._ref_z = 0.3
        self._ep_start_time = perf_counter()
        self._ep_count = 0
        self._steps = 0
        self._step_time_prev = perf_counter()
        self.dt_ema = 0.0
        self.alpha_ema = 0.1
        self._vx_errors = []
        self._wz_errors = []

    def _nanosec_now(self):
        return self.node.get_clock().now().nanoseconds

    def _get_obs(self):
        self.node.update_tf()

        lin_vel_norm = norm11(
            self.node.real_lin_vel, self.lin_vel_low, self.lin_vel_high
        )
        ang_vel_norm = norm11(
            self.node.real_ang_vel, -self.ang_vel_max, self.ang_vel_max
        )
        ori_norm = norm11(self.node.real_orientation[:2], -pi, pi)

        joint_pos_norm = norm11(self.node.joint_positions, jpos_low, jpos_high)
        joint_vel_norm = norm11(self.node.joint_velocities, -jvel_max, jvel_max)

        ref_vel_norm = norm11(self._ref_vel, -self.ref_vel_max, self.ref_vel_max)
        ref_z_norm = norm01(
            np.array([self._ref_z], dtype=float32), self.ref_z_low, self.ref_z_high
        )

        obs = {
            "linear_vel": lin_vel_norm,
            "angular_vel": ang_vel_norm,
            "orientation": ori_norm,
            "joint_positions": joint_pos_norm,
            "joint_velocities": joint_vel_norm,
            "ref_vel": ref_vel_norm,
            "ref_z": ref_z_norm,
            "prev_action": self._prev_action.copy(),
        }

        self.tb_log["obs/lin_vel_x"] = lin_vel_norm[0]
        self.tb_log["obs/lin_vel_y"] = lin_vel_norm[1]
        self.tb_log["obs/lin_vel_z"] = lin_vel_norm[2]
        self.tb_log["obs/ang_vel_x"] = ang_vel_norm[0]
        self.tb_log["obs/ang_vel_y"] = ang_vel_norm[1]
        self.tb_log["obs/ang_vel_z"] = ang_vel_norm[2]
        self.tb_log["obs/roll"] = ori_norm[0]
        self.tb_log["obs/pitch"] = ori_norm[1]
        self.tb_log["obs/ref_z"] = ref_z_norm[0]

        for i, val in enumerate(joint_pos_norm):
            self.tb_log[f"obs/joint_pos_{i}"] = val
        for i, val in enumerate(joint_vel_norm):
            self.tb_log[f"obs/joint_vel_{i}"] = val
        for i, val in enumerate(self._prev_action):
            self.tb_log[f"obs/prev_action_{i}"] = val

        return obs

    def _get_info(self):
        return {
            "real_position": self.node.real_position,
            "real_orientation": self.node.real_orientation,
            "real_linear_vel": self.node.real_lin_vel,
            "real_angular_vel": self.node.real_ang_vel,
        }

    def _is_done(self, obs, info) -> Done:
        ori = info["real_orientation"]
        z = info["real_position"][2]
        lv = info["real_linear_vel"]
        av = info["real_angular_vel"]
        roll = ori[0]
        pitch = ori[1]
        ep_time = (perf_counter() - self._ep_start_time) * self.rtf
        max_vx = self._ref_vel[0] * 2
        max_vy = self._ref_vel[1] * 2
        max_wz = self._ref_vel[2] * 2

        if (
            abs(roll) > self.roll_th
            or abs(pitch) > self.pitch_th
            or z <= self.z_min_th
            or abs(lv[0]) > max_vx
            or abs(lv[1]) > max_vy
            or abs(av[2]) > max_wz
        ):
            return Done.TERMINATED
        elif ep_time >= self.max_ep_time:
            return Done.TRUNCATED
        else:
            return Done.ALIVE

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.node.publish_position_commands(self.q_homing)

        # TODO: Make start pose more diverse
        roll = 0
        pitch = 0
        yaw = self.np_random.uniform(-pi, pi)

        pose = Pose()
        pose.position.z = 0.24
        pose.orientation = quat_msg_from_euler(roll, pitch, yaw)

        pose_base_foot = Pose()
        pose_base_foot.position.z = pose.position.z
        pose_base_foot.orientation = quat_msg_from_euler(roll, pitch, 0)

        pose_foot_odom = Pose()
        pose_foot_odom.position.x = pose.position.x
        pose_foot_odom.position.y = pose.position.y
        pose_foot_odom.orientation = quat_msg_from_euler(0, 0, yaw)

        # TODO: Make service calls prettier
        wait_for_all_futures(
            self.executor,
            [
                ServiceCall(
                    "Robot reset service",
                    self.node.send_reset_request(pose, self.q_homing),
                    bool_res_handler,
                ),
                ServiceCall(
                    "Base to footprint EKF set pose service",
                    self.node.send_base_to_foot_ekf_set_pose_request(pose_base_foot),
                    None,
                ),
                ServiceCall(
                    "Footprint to odom EKF set pose service",
                    self.node.send_foot_to_odom_ekf_set_pose_request(pose_foot_odom),
                    None,
                ),
            ],
            self.node.get_logger(),
            self.reset_srv_timeout,
        )

        while not self.node.tf_buffer.can_transform(
            self.node.target_frame,
            self.node.source_frame,
            Time(),
            Duration(seconds=0.01),
        ):
            self.node.get_logger().info(
                f"Waiting for tf {self.node.source_frame} -> {self.node.target_frame} to become available",
                throttle_duration_sec=5,
            )
            self.executor.spin_once(0.01)

        # Grid Adaptive Curriculum
        active_indices = np.where(self.active_mask)
        idx = self.np_random.integers(len(active_indices[0]))
        ix, iw = active_indices[0][idx], active_indices[1][idx]
        self._ref_vel[0] = self.vx_grid[ix]
        self._ref_vel[2] = self.wz_grid[iw]
        self._ref_vel[1] = self.np_random.uniform(-self.ref_vy_max, self.ref_vy_max)
        self._ref_z = self.np_random.uniform(self.ref_z_low, self.ref_z_high)

        if len(self._vx_errors):
            vx_err_mean = sum(self._vx_errors) / len(self._vx_errors)
            wz_err_mean = sum(self._wz_errors) / len(self._wz_errors)
        else:
            vx_err_mean = 2
            wz_err_mean = 2

        self._vx_errors.clear()
        self._wz_errors.clear()
        self._ep_count += 1
        self._prev_action = np.zeros((12,), dtype=float32)

        self.tb_log["episode/time"] = (perf_counter() - self._ep_start_time) * self.rtf
        self.tb_log["episode/count"] = self._ep_count
        self.tb_log["episode/vx_err_mean"] = vx_err_mean
        self.tb_log["episode/wz_err_mean"] = wz_err_mean
        self.tb_log["episode/cmd_vx"] = self._ref_vel[0]
        self.tb_log["episode/cmd_vy"] = self._ref_vel[1]
        self.tb_log["episode/cmd_wz"] = self._ref_vel[2]

        sleep(self.reset_delay)
        self.executor.spin_once(0.01)

        self._ep_start_time = perf_counter()
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        action_total = np.clip(self.q_homing_normalized + action, -1, 1)
        q = denorm11(action_total, jpos_low, jpos_high)

        self.node.publish_position_commands(q)
        self.executor.spin_once(timeout_sec=0.01)

        obs = self._get_obs()
        info = self._get_info()
        done = self._is_done(obs, info)

        terminated = True if done == Done.TERMINATED else False
        truncated = True if done == Done.TRUNCATED else False

        action_rate = np.linalg.norm(action - obs["prev_action"])
        pose_unsimilarity = np.linalg.norm(action_total - self.q_homing_normalized)
        z_err = obs["ref_z"][0] - norm01(
            info["real_position"][2], self.ref_z_low, self.ref_z_high
        )
        wz_err = obs["ref_vel"][2] - obs["angular_vel"][2]
        xy_vel_err = np.linalg.norm(obs["ref_vel"][:2] - obs["linear_vel"][:2])
        z_spd = obs["linear_vel"][2]
        roll = obs["orientation"][0]
        pitch = obs["orientation"][1]
        dt = (perf_counter() - self._step_time_prev) * self.rtf
        if dt > 0:
            self.dt_ema = self.alpha_ema * dt + (1 - self.alpha_ema) * self.dt_ema

        reward = (
            exp(-(xy_vel_err**2 * 2)) * 1.5 * self.dt_ema
            + exp(-(wz_err**2 * 2)) * 1.0 * self.dt_ema
            - z_spd**2 * 4 * self.dt_ema  # 0.16
            - (roll**2 + pitch**2) * self.dt_ema
            - action_rate**2 * 0.2 * self.dt_ema  # 2.5
            - pose_unsimilarity**2 * 0.2 * self.dt_ema  # 1
            - z_err**2 * self.dt_ema  # 0.04
        )
        reward = np.clip(reward, -10.0, 10.0)

        self._prev_action = action.copy()
        self._step_time_prev = perf_counter()
        self._vx_errors.append(abs(obs["ref_vel"][0] - obs["linear_vel"][0]))
        self._wz_errors.append(abs(obs["ref_vel"][2] - obs["linear_vel"][2]))

        # Log values
        self.tb_log["step/action_rate"] = action_rate
        self.tb_log["step/pose_unsimilarity"] = pose_unsimilarity
        self.tb_log["step/errors/z"] = z_err
        self.tb_log["step/errors/wz"] = wz_err
        self.tb_log["step/errors/xy_vel"] = xy_vel_err
        self.tb_log["step/z_vel"] = obs["linear_vel"][2]
        self.tb_log["step/orientation/roll"] = obs["orientation"][0]
        self.tb_log["step/orientation/pitch"] = obs["orientation"][1]
        self.tb_log["step/dt"] = dt
        self.tb_log["step/dt_ema"] = self.dt_ema

        # Throttle step freq when no learning
        if self._steps < self.learning_starts:
            self._steps += 1
            sleep(self.step_delay)

        return obs, reward, terminated, truncated, info

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()

    def print_debug_state(self):
        if self._steps % 10 == 0:
            self.node.print_debug_state()
            lines = [
                f"Ref Z   : {self._ref_z}",
                f"Ref VX  : {self._ref_vel[0]}",
                f"Ref VY  : {self._ref_vel[1]}",
                f"Ref WZ  : {self._ref_vel[2]}",
            ]
            print("\n".join(lines))
            # print(self._get_obs())  # TODO: Pretty obs print
