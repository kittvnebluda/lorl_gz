from argparse import ArgumentParser

from numpy.random import randint
import rclpy
import torch
from legged_rl_env.go2_env import Go2Env
from legged_rl_env.gym_wrappers import FlattenObsDict
from legged_rl_env.training import ALGOS
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.utils import set_random_seed

assert torch.xpu.is_available(), "XPU not available"
assert torch.xpu.device_count() > 0, "No XPU devices found"

MODEL_DIR = "./models/"
TENSORBOARD_LOG_DIR = "./tensorboard/"
DEVICE = "xpu"

random_seed = randint(0, 2**32 - 1)


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        step_log_mean = {}

        envs = self.training_env.envs  # pyright: ignore[reportAttributeAccessIssue]
        n_envs = len(envs)

        # Accumulate
        for env in envs:
            sl = env.unwrapped.step_log
            for key, value in sl.items():
                step_log_mean[key] = step_log_mean.get(key, 0.0) + float(value)

        # Average and record
        for key in step_log_mean:
            step_log_mean[key] /= n_envs
            self.logger.record(f"step/{key}", step_log_mean[key])

        return True


class HParamCallback(BaseCallback):
    def __init__(self, params, verbose: int = 0):
        super().__init__(verbose)
        self.params = params

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "model_seed": int(random_seed),
        }
        hparam_dict.update(self.params)

        metric_dict = {
            "step/ep_time": 0.0,
            "step/errors/xy_vel": 0.0,
            "step/errors/wz": 0.0,
            "step/errors/z": 0.0,
            "step/orientation/pitch": 0.0,
            "step/orientation/roll": 0.0,
            "train/actor_loss": 0.0,
            "train/critic_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


def main(args):
    set_random_seed(random_seed)

    env = FlattenObsDict(Go2Env())
    check_env(env)

    logger = env.unwrapped.node.get_logger()  # pyright: ignore[reportAttributeAccessIssue]

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
        model = algo_cls.load(model_name, **params)
    else:
        print(f"Creating new model {model_name}")
        model = algo_cls(**params)

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=[
            TensorboardCallback(),
            HParamCallback(cfg["params"]),
            checkpoint_callback,
        ],
    )
    model.save(MODEL_DIR + model_save_name)

    logger.info("Training finished")

    vec_env = model.get_env()
    if vec_env is None:
        logger.info("Vec Env is None")
        return

    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    logger.info(f"Reward mean: {mean_reward}, std: {std_reward}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--load_model",
        "-l",
        type=str,
        default="",
        help="Name of the file to load",
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
    try:
        main(args)
    except KeyboardInterrupt:
        pass
