from argparse import ArgumentParser
from time import sleep

import rclpy
import torch
from legged_rl_env.go2_env import Go2Env
from legged_rl_env.gym_wrappers import FlattenObsDict
from legged_rl_env.training import ALGOS
from rclpy.node import Node
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

assert torch.xpu.is_available(), "XPU not available"
assert torch.xpu.device_count() > 0, "No XPU devices found"

MODEL_DIR = "./models/"
TENSORBOARD_LOG_DIR = "./tensorboard/"
DEVICE = "xpu"


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


class SimSyncCallback(BaseCallback):
    def __init__(self, node: Node, target_dt=0.01, verbose=0):
        super().__init__(verbose)
        self.node = node
        self.target_dt = target_dt
        self.target_dt_with_reserve = self.target_dt * 0.8
        self.last_time = self.node.get_clock().now()

    def _on_step(self) -> bool:
        current_time = self.node.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds
        if dt <= self.target_dt_with_reserve:
            sleep(self.target_dt - dt)
        self.last_time = current_time
        return True


def main(args):
    env = FlattenObsDict(Go2Env())
    check_env(env)

    if args.load_model:
        algo_name = args.load_model.split("_")[-1].replace(".zip", "")
        model_name = args.load_model.replace(".zip", "")
    else:
        algo_name = args.algo.lower()
        model_name = f"{args.name}_{args.algo}"

    model_save_name = f"{args.name}_{algo_name}"

    tensorboard_log_path = TENSORBOARD_LOG_DIR + model_save_name + "/"
    print(
        f"Using {algo_name.upper()}; Model name: {model_save_name}; Tensorboard logs: {tensorboard_log_path}"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix=model_save_name,
    )

    cfg = ALGOS[algo_name]
    algo_cls = cfg["class"]
    params = cfg["params"].copy()

    params["env"] = env
    params["tensorboard_log"] = tensorboard_log_path
    params["device"] = DEVICE

    if args.load_model:
        print(f"Loading model {model_name}")
        model = algo_cls.load(MODEL_DIR + model_name, **params)
    else:
        print(f"Creating new model {model_name}")
        model = algo_cls(**params)

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=[
            TensorboardCallback(),
            checkpoint_callback,
            # SimSyncCallback(env.unwrapped.node),
        ],
    )
    model.save(MODEL_DIR + model_save_name)

    env.unwrapped.node.get_logger().info("Training finished")

    vec_env = model.get_env()
    if vec_env is None:
        env.unwrapped.node.get_logger().info("Vec Env is None")
        return

    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    env.unwrapped.node.get_logger().info(
        f"Reward mean: {mean_reward}, std: {std_reward}"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--load_model",
        "-l",
        type=str,
        default="",
        help=f"Name of the file to load from {MODEL_DIR} directory",
    )
    parser.add_argument("--total_timesteps", "-t", type=int, default=50000)
    parser.add_argument(
        "--name", "-n", type=str, default="go2", help="Name for a model file"
    )
    parser.add_argument(
        "--algo",
        "-a",
        type=str,
        default="ppo",
        help=f"Choose algorithm to train: {list(ALGOS.keys())}, will be added to the end of a model file name",
    )
    args = parser.parse_args()

    assert len(args.name) != 0 and len(args.algo) != 0, "Invalid arguments"

    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    main(args)
