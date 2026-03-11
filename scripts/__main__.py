from argparse import ArgumentParser

import rclpy

from .random_agent import random_agent
from .reset_agent import reset_agent
from .run_agent import run_agent
from .train import ALGOS, train

parser = ArgumentParser()
subparser = parser.add_subparsers(required=True)

# ------------- TRAIN -------------

train_parser = subparser.add_parser("train")
train_parser.add_argument(
    "--load_model",
    "-l",
    type=str,
    default="",
    help="Name of the file to load",
)
train_parser.add_argument("--total_timesteps", "-t", type=int, default=50000)
train_parser.add_argument(
    "--name", "-n", type=str, default="go2", help="Name for a model file"
)
train_parser.add_argument(
    "--algo",
    "-a",
    type=str,
    default="ppo",
    help=f"Choose algorithm to train: {list(ALGOS.keys())}, will be added to the end of a model file name",
)
train_parser.add_argument("--step_delay", type=float, default=0)
train_parser.add_argument("--reset_delay", type=float, default=0)
train_parser.add_argument("--real_time_factor", type=float, default=1)
train_parser.set_defaults(factory=lambda args: train(args))

# ------------- RUN -------------

run_parser = subparser.add_parser("run")
run_subparser = run_parser.add_subparsers(required=True)

run_model_parser = run_subparser.add_parser("model")
run_model_parser.add_argument("model", type=str)
run_model_parser.add_argument("--algo", "-a", type=str)
run_model_parser.add_argument("--debug", "-d", action="store_true")
run_model_parser.set_defaults(factory=lambda args: run_agent(args))

random_agent_parser = run_subparser.add_parser("random")
random_agent_parser.add_argument("--debug", "-d", action="store_true")
random_agent_parser.set_defaults(factory=lambda args: random_agent(args))

# ------------- RESET -------------

reset_parser = subparser.add_parser("reset")
reset_parser.set_defaults(factory=lambda _: reset_agent())

args = parser.parse_args()

rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
args.factory(args)
