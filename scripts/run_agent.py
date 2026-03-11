from time import time

from legged_rl_env.go2_env import Go2Env
from legged_rl_env.gym_wrappers import FlattenObsDict
from numpy import mean

from .train import ALGOS


def run_agent(args):
    assert args.model, "Invalid model argument"

    env = FlattenObsDict(Go2Env(learning=False))

    algo_name = args.model.split("_")[-1].replace(".zip", "")
    try:
        cfg = ALGOS[algo_name]
    except KeyError:
        cfg = ALGOS[args.algo]
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
                env.unwrapped.print_debug_state()

            if dones[0]:
                time_of_lifes.append(time() - start_time)
                start_time = time()

    except KeyboardInterrupt:
        print()
        pass

    print(f"Min time of life: {min(time_of_lifes):2f}s")
    print(f"Avg time of life: {mean(time_of_lifes):.2f}s")
    print(f"Max time of life: {max(time_of_lifes):2f}s")
