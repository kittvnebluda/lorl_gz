from legged_rl_env.go2_env import Go2Env
import rclpy


def main():
    env = Go2Env()
    env.reset()
    env.node.get_logger().info("Agent reset")


if __name__ == "__main__":
    rclpy.init()
    main()
