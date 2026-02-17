from enum import Enum, auto

import gymnasium as gym
import numpy as np
import rclpy.executors
from geometry_msgs.msg import Pose
from gymnasium import spaces
from numpy import deg2rad, exp, float32, pi
from rclpy.duration import Duration
from rclpy.time import Time
from tf_transformations import quaternion_from_euler

from legged_rl_env.constants import (
    jpos_high,
    jpos_low,
    jvel_max,
    stable_standing_joint_positions,
)
from legged_rl_env.go2_node import Go2Node
from legged_rl_env.utils import ServiceCall, bool_res_handler, wait_for_all_futures


class Done(Enum):
    ALIVE = auto()
    TERMINATED = auto()
    TRUNCATED = auto()


def normalize(x, low, high):
    return 2.0 * (x - low) / (high - low) - 1.0


def denormalize(x_norm, low, high):
    return low + (x_norm + 1.0) * 0.5 * (high - low)


class Go2Env(gym.Env):
    metadata = {}
    reset_srv_timeout = 1.0
    q_homing = np.array(
        stable_standing_joint_positions, dtype=float32
    )  # Default joint position
    q_homing_normalized = normalize(q_homing, jpos_low, jpos_high).astype(float32)

    def __init__(self, wait_for_services=True, render_mode=None) -> None:
        self.render_mode = render_mode

        self.observation_space = spaces.Dict(
            {
                "linear_vel": spaces.Box(
                    low=np.array([-3.5, -2.0, -1.0], dtype=float32),
                    high=np.array([3.7, 2.0, 1.0], dtype=float32),
                    shape=(3,),
                    dtype=float32,
                ),
                "angular_vel": spaces.Box(
                    low=np.array([-4.0, -4.0, -3.0], dtype=float32),
                    high=np.array([4.0, 4.0, 3.0], dtype=float32),
                    shape=(3,),
                    dtype=float32,
                ),
                "orientation": spaces.Box(low=-pi, high=pi, shape=(2,), dtype=float32),
                "joint_positions": spaces.Box(
                    low=jpos_low, high=jpos_high, shape=(12,), dtype=float32
                ),
                "joint_velocities": spaces.Box(
                    low=-jvel_max, high=jvel_max, shape=(12,), dtype=float32
                ),
                "ref_vel": spaces.Box(
                    low=np.array([-2, -2, -1], dtype=float32),
                    high=np.array([2, 2, 1], dtype=float32),
                    shape=(3,),
                    dtype=float32,
                ),
                "ref_z": spaces.Box(low=0.0, high=0.5, shape=(1,), dtype=float32),
                "prev_action": spaces.Box(low=-1, high=1, shape=(12,), dtype=float32),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=float32)

        self.roll_th = deg2rad(45)
        self.pitch_th = deg2rad(45)
        self.z_min_th = 0.15
        self.max_steps = 1000

        self.action_diff_magnitude = 0

        self._prev_action = self.q_homing_normalized
        self._curr_step = 0
        self._ref_vel = np.array([0.0, 0.0, 0.0], dtype=float32)  # vx,vy,wz
        self._ref_z = 0.3

        self.node = Go2Node(wait_for_services=wait_for_services)
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def _get_obs(self):
        self.node.update_pose()
        return {
            "linear_vel": self.node.base_linear_vel,
            "angular_vel": self.node.base_angular_vel,
            "orientation": self.node.base_orientation[:2],
            "joint_positions": self.node.joint_positions,
            "joint_velocities": self.node.joint_velocities,
            "prev_action": self._prev_action,
            "ref_vel": self._ref_vel,
            "ref_z": np.array([self._ref_z], dtype=np.float32),
        }

    def _get_info(self):
        return {
            "position": self.node.real_position,
            "orientation": self.node.real_orientation,
        }

    def _is_done(self, obs, info, curr_step) -> Done:
        ori = info["orientation"]
        roll = ori[0]
        pitch = ori[1]
        z = info["position"][2]

        if abs(roll) > self.roll_th or abs(pitch) > self.pitch_th or z <= self.z_min_th:
            return Done.TERMINATED
        elif curr_step >= self.max_steps:
            return Done.TRUNCATED
        else:
            return Done.ALIVE

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.node.publish_position_commands(self.q_homing)

        x, y, z, w = quaternion_from_euler(0, 0, self.np_random.uniform(-pi, pi))

        pose = Pose()
        pose.position.z = 0.24
        pose.orientation.w = w
        pose.orientation.x = x
        pose.orientation.y = y
        pose.orientation.z = z

        pose_base_foot = Pose()
        pose_base_foot.position.z = pose.position.z
        pose_base_foot.orientation = pose.orientation

        pose_foot_odom = Pose()
        pose_foot_odom.position.x = pose.position.x
        pose_foot_odom.position.y = pose.position.y

        futures = [
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
        ]
        wait_for_all_futures(self.executor, futures, self.node.get_logger())

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

        self._ref_z = self.np_random.uniform(0.25, 0.35)
        self._ref_vel[:2] = self.np_random.uniform(-2, 2, (2,))
        self._ref_vel[2] = self.np_random.uniform(-1, 1)

        obs = self._get_obs()
        info = self._get_info()

        self._prev_action = self.q_homing_normalized
        self._curr_step = 0

        return obs, info

    def step(self, action):
        action_total = np.clip(self.q_homing_normalized + action, -1, 1)
        q = denormalize(action_total, jpos_low, jpos_high)
        self.node.publish_position_commands(q)

        self.executor.spin_once(timeout_sec=0.01)

        obs = self._get_obs()
        info = self._get_info()
        done = self._is_done(obs, info, self._curr_step)

        terminated = True if done == Done.TERMINATED else False
        truncated = True if done == Done.TRUNCATED else False

        reward = (
            exp(-(np.linalg.norm(obs["ref_vel"][:2] - obs["linear_vel"][:2]) ** 2))
            + exp(-((obs["ref_vel"][2] - obs["angular_vel"][2]) ** 2))
            - (obs["ref_z"][0] - info["position"][2]) ** 2
            - np.linalg.norm(q - self.q_homing) ** 2  # Pose similarity
            - np.linalg.norm(action - obs["prev_action"]) ** 2  # Action rate penalty
            - obs["linear_vel"][2] ** 2
            - (info["orientation"][0] ** 2 + info["orientation"][1] ** 2)
        )

        self.action_diff_magnitude = np.linalg.norm(action - obs["prev_action"])

        self._prev_action = action
        self._curr_step += 1

        return obs, reward, terminated, truncated, info

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()
