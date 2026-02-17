from argparse import ArgumentParser
from time import time

import rclpy
from legged_rl_env.go2_env import Go2Env
from legged_rl_env.gym_wrappers import FlattenObsDict
from legged_rl_env.training import ALGOS
from numpy import mean


def main(args):
    env = FlattenObsDict(Go2Env())

    algo_name = args.model.split("_")[-1].replace(".zip", "")
    cfg = ALGOS[algo_name]
    algo_cls = cfg["class"]
    params = cfg["params"].copy()
    params["env"] = env

    model = algo_cls.load(args.model, **params)

    vec_env = model.get_env()
    if vec_env is None:
        raise RuntimeError("VecEnv is None")

    obs = vec_env.reset()
    start_time = time()
    time_of_lifes = []

    try:
        while 1:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            if args.debug:
                env.unwrapped.node.print_debug_state()

            if dones[0]:
                time_of_lifes.append(time() - start_time)
                start_time = time()

    except KeyboardInterrupt:
        pass

    print(f"Mean time of life: {mean(time_of_lifes):.2f}s")
    print(f"Min time of life: {min(time_of_lifes):2f}s")
    print(f"Max time of life: {max(time_of_lifes):2f}s")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    if len(args.model) == 0:
        raise ValueError("Invalid model argument")

    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    main(args)
