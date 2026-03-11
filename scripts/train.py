from argparse import ArgumentParser

import numpy as np
import rclpy
import torch
from legged_rl_env.go2_env import Go2Env
from legged_rl_env.gym_wrappers import FlattenObsDict
from legged_rl_env.training import ALGOS
from numpy.random import randint
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.utils import set_random_seed

assert torch.xpu.is_available(), "XPU not available"
assert torch.xpu.device_count() > 0, "No XPU devices found"

MODEL_DIR = "./models/"
TENSORBOARD_LOG_DIR = "./tensorboard/yay/"
DEVICE = "xpu"

random_seed = randint(0, 2**32 - 1)


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        tl_mean = {}

        envs = self.training_env.envs  # pyright: ignore[reportAttributeAccessIssue]
        n_envs = len(envs)

        # Accumulate
        for env in envs:
            tl = env.unwrapped.tb_log
            for key, value in tl.items():
                tl_mean[key] = tl_mean.get(key, 0.0) + float(value)
            tl.clear()

        # Average and record
        for key in tl_mean:
            tl_mean[key] /= n_envs
            self.logger.record(key, tl_mean[key])

        return True


class HParamCallback(BaseCallback):
    def __init__(self, params, rtf, verbose: int = 0):
        super().__init__(verbose)
        self.params = params
        self.rtf = rtf

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "model_seed": int(random_seed),
            "rtf": self.rtf,
        }
        hparam_dict.update(self.params)

        metric_dict = {
            "episode/time": 0.0,
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


class GridAdaptiveCurriculumCallback(BaseCallback):
    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if "episode" not in self.locals:
            return

        v_err = self.locals["episode"]["vx_err_mean"]
        w_err = self.locals["episode"]["wz_err_mean"]
        if v_err < self.threshold and w_err < self.threshold:
            envs = self.training_env.envs  # pyright: ignore[reportAttributeAccessIssue]
            n_envs = len(envs)
            assert n_envs == 1
            env = envs[0]

            cmd_v_x = self.locals["episode"]["cmd_vx"]
            cmd_w_z = self.locals["episode"]["cmd_wz"]

            idx_x = np.argmin(np.abs(env.vx_grid - cmd_v_x))
            idx_w = np.argmin(np.abs(env.wz_grid - cmd_w_z))

            for dx, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, nw = idx_x + dx, idx_w + dw
                if (
                    0 <= nx < env.active_mask.shape[0]
                    and 0 <= nw < env.active_mask.shape[1]
                ):
                    env.active_mask[nx, nw] = True


def main(args):
    set_random_seed(random_seed)

    if args.load_model:
        algo_name = args.load_model.split("_")[-1].replace(".zip", "")
        model_name = args.load_model.replace(".zip", "")
    else:
        algo_name = args.algo.lower()
        model_name = f"{args.name}_{args.algo}"

    model_save_name = f"{args.name}_{algo_name}"

    cfg = ALGOS[algo_name]
    algo_cls = cfg["class"]
    params = cfg["params"].copy()

    env = FlattenObsDict(
        Go2Env(
            step_delay=args.step_delay,
            reset_delay=args.reset_delay,
            real_time_factor=args.real_time_factor,
            learning_starts=params["learning_starts"],
        )
    )
    check_env(env)

    logger = env.unwrapped.node.get_logger()  # pyright: ignore[reportAttributeAccessIssue]

    print(f"Using {algo_name.upper()}; Model name: {model_save_name}")

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix=model_save_name,
    )

    params["env"] = env
    params["device"] = DEVICE
    params["tensorboard_log"] = TENSORBOARD_LOG_DIR

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
            HParamCallback(cfg["params"], args.real_time_factor),
            checkpoint_callback,
            GridAdaptiveCurriculumCallback(),
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
    parser.add_argument("--step_delay", type=float, default=0)
    parser.add_argument("--reset_delay", type=float, default=0)
    parser.add_argument("--real_time_factor", type=float, default=1)
    args = parser.parse_args()

    assert len(args.name) != 0 and len(args.algo) != 0, "Invalid arguments"

    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    try:
        main(args)
    except KeyboardInterrupt:
        pass
