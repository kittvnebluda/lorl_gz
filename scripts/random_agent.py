from argparse import ArgumentParser

import rclpy
from gymnasium.utils.env_checker import check_env

from legged_rl_env.go2_env import Go2Env


def main(args):
    env = Go2Env()

    obs, info = env.reset()
    for k, v in obs.items():
        space = env.observation_space[k]  # pyright: ignore[reportIndexIssue]

        if not space.contains(v):
            env.node.get_logger().error(f"FAILED: {k}")
            env.node.get_logger().error(f"value: {v}")
            env.node.get_logger().error(f"space: {space}")

    try:
        check_env(env, skip_render_check=True)
    except Exception as e:
        env.node.get_logger().warn(f"Exception in check_env: {e}")

    try:
        while 1:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if args.debug:
                env.print_debug_state()
            if done:
                env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    main(args)
