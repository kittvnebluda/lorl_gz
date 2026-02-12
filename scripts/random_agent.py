import rclpy
from gymnasium.utils.env_checker import check_env

from legged_rl_env.go2_env import Go2Env


def main():
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

    for _ in range(5):
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            total_reward += reward

        print("Episode reward:", total_reward)
        env.reset()

    env.close()


if __name__ == "__main__":
    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    main()
