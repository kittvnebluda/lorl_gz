from argparse import ArgumentParser

import rclpy
from legged_rl_env.go2_env import Go2Env
from legged_rl_env.gym_wrappers import FlattenObsDict
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import PPO


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        action_magnitudes = []
        for env in self.training_env.envs:
            action_magnitudes.append(env.unwrapped.action_magnitude)

        self.logger.record(
            "action_magnitude", sum(action_magnitudes) / len(action_magnitudes)
        )
        return True


def main(args):
    env = FlattenObsDict(Go2Env())
    check_env(env)

    if args.load_model:
        model = PPO.load(
            "go2", env, tensorboard_log="./ppo_go2_tensorboard/", verbose=0
        )
    else:
        model = PPO(
            "MlpPolicy", env, tensorboard_log="./ppo_go2_tensorboard/", verbose=0
        )

    try:
        model.learn(
            total_timesteps=args.steps_num,
            progress_bar=True,
            callback=TensorboardCallback(),
        )
    except KeyboardInterrupt:
        pass
    finally:
        model.save("go2")

    env.env.node.get_logger().info("Training finished")

    vec_env = model.get_env()
    if vec_env is None:
        env.env.node.get_logger().info("Vec Env is None")
        return

    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    env.env.node.get_logger().info(f"Reward mean: {mean_reward}, std: {std_reward}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_model", "-l", action="store_true")
    parser.add_argument("--steps_num", "-s", type=int, default=50000)
    args = parser.parse_args()

    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    main(args)
