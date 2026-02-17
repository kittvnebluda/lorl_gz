import numpy as np
import rclpy
import rclpy.time
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry
from numpy.typing import NDArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.task import Future
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64MultiArray
from tf2_ros import TransformException  # pyright: ignore[reportAttributeAccessIssue]
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import euler_from_quaternion

from legged_rl_env.constants import joint_names, jpos_low, jvel_max
from legged_rl_interfaces.srv import ResetRobot


class Go2Node(Node):
    def __init__(self, wait_for_services=True) -> None:
        super().__init__("go2_env")

        self.base_linear_vel = np.zeros(3, dtype=np.float32)
        self.base_angular_vel = np.zeros(3, dtype=np.float32)
        self.base_position = np.zeros(3, dtype=np.float32)
        self.base_orientation = np.zeros(3, dtype=np.float32)
        self.joint_positions = jpos_low
        self.joint_velocities = -jvel_max
        self.ref_z = np.float32(0)
        self.real_position = np.zeros(3, dtype=np.float32)
        self.real_orientation = np.zeros(3, dtype=np.float32)

        fast_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, fast_qos
        )
        self.odom_local_sub = self.create_subscription(
            Odometry, "/odom/local", self.odom_local_callback, fast_qos
        )
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_states_callback, fast_qos
        )
        self.ref_z_sub = self.create_subscription(
            Float32, "/ref_z", self.ref_z_callback, fast_qos
        )
        self.real_pose_sub = self.create_subscription(
            PoseStamped, "/real_pose", self.real_pose_callback, fast_qos
        )
        self.position_commands_pub = self.create_publisher(
            Float64MultiArray,
            "/position_controller/commands",
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                durability=DurabilityPolicy.VOLATILE,
                deadline=Duration(seconds=0.02),
            ),
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.source_frame = "map"
        self.target_frame = "base_link"
        self.reset_req = ResetRobot.Request()

        self.resetter = self.create_client(ResetRobot, "reset_robot")
        while wait_for_services and not self.resetter.wait_for_service(5.0):
            self.get_logger().warn(
                "Reset robot service not available, waiting again..."
            )

    # From /odom (odom -> base_footprint)
    def odom_callback(self, msg: Odometry):
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular

        self.base_linear_vel[0] = lv.x
        self.base_linear_vel[1] = lv.y
        self.base_angular_vel[2] = av.z

    # From /odom/local (base_footprint -> base_link)
    def odom_local_callback(self, msg: Odometry):
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular

        self.base_linear_vel[2] = lv.z
        self.base_angular_vel[0] = av.x
        self.base_angular_vel[1] = av.y

    def joint_states_callback(self, msg: JointState):
        if len(msg.name) != len(msg.position) or len(msg.name) != len(msg.velocity):
            self.get_logger().error("Malformed JointState: lengths mismatch")
            return
        if len(msg.position) != 12 or len(msg.velocity) != 12:
            self.get_logger().error("Joint states does not have 12 joints")
            return

        for i in range(12):
            idx = joint_names.index(msg.name[i])  # pyright: ignore[reportIndexIssue]
            self.joint_positions[i] = msg.position[idx]
            self.joint_velocities[i] = msg.velocity[idx]

    def ref_z_callback(self, msg: Float32):
        self.ref_z = np.float32(msg.data)

    def real_pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        r = msg.pose.orientation
        euler = euler_from_quaternion((r.x, r.y, r.z, r.w))
        self.real_position = np.array([p.x, p.y, p.z], dtype=np.float32)
        self.real_orientation = np.array(euler, dtype=np.float32)

    def update_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.target_frame, self.source_frame, rclpy.time.Time()
            )
        except TransformException as e:
            self.get_logger().warn(
                f"Could not transform from {self.source_frame} to {self.target_frame}: {e}. Exception type: {type(e)}"
            )
            return

        p = t.transform.translation
        r = t.transform.rotation
        euler = euler_from_quaternion((r.x, r.y, r.z, r.w))
        self.base_orientation = np.array(euler, dtype=np.float32)
        self.base_position = np.array([p.x, p.y, p.z], dtype=np.float32)

    def send_reset_request(
        self, pose: Pose, joint_positions: tuple[float, ...] | NDArray
    ) -> Future:
        self.reset_req.pose = pose
        self.reset_req.joint_positions = joint_positions
        self.reset_req.joint_names = joint_names
        return self.resetter.call_async(self.reset_req)

    def publish_position_commands(self, positions: tuple[float, ...] | NDArray):
        if len(positions) != 12:
            self.get_logger().error("Position commands number does not equal to 12")
            return
        self.position_commands_pub.publish(Float64MultiArray(data=positions))


def main():
    rclpy.init()
    node = Go2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
