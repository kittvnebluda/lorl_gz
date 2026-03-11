import torch
from legged_rl_env.go2_env import Go2Env
from legged_rl_env.gym_wrappers import FlattenObsDict
from numpy.random import randint
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

from .sb3_callbacks import (
    GridAdaptiveCurriculumCallback,
    HParamCallback,
    TensorboardCallback,
)

ALGOS = {
    "ppo": {
        "class": PPO,
        "params": dict(
            policy="MlpPolicy",
        ),
    },
    "sac": {
        "class": SAC,
        "params": dict(
            policy="MlpPolicy",
            learning_rate=3e-4,
            buffer_size=2_000_000,
            learning_starts=50_000,
            batch_size=4096,
            tau=0.005,
            gamma=0.99,
            train_freq=20,
            gradient_steps=1,
            ent_coef="auto",
            target_entropy="auto",
        ),
    },
}

assert torch.xpu.is_available(), "XPU not available"
assert torch.xpu.device_count() > 0, "No XPU devices found"


def train(args):
    assert len(args.name) != 0 and len(args.algo) != 0, "Invalid arguments"

    models_dir = "./models/"
    tensorboard_log_dir = "./tensorboard/yay/"
    device = "xpu"

    random_seed = randint(0, 2**32 - 1)

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
    params["device"] = device
    params["tensorboard_log"] = tensorboard_log_dir

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
            HParamCallback(cfg["params"], args.real_time_factor, random_seed),
            checkpoint_callback,
            GridAdaptiveCurriculumCallback(),
        ],
    )
    model.save(models_dir + model_save_name)

    logger.info("Training finished")

    vec_env = model.get_env()
    if vec_env is None:
        logger.info("Vec Env is None")
        return

    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    logger.info(f"Reward mean: {mean_reward}, std: {std_reward}")
