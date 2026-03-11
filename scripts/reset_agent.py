from legged_rl_env.go2_env import Go2Env


def reset_agent():
    env = Go2Env()
    env.reset()
    env.node.get_logger().info("Agent reset")
