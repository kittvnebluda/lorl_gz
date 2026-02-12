from legged_rl_env.gym_wrappers import FlattenObsDict
import rclpy
from legged_rl_env.go2_env import Go2Env
from stable_baselines3 import PPO


def main():
    env = FlattenObsDict(Go2Env())
    model = PPO.load("go2", env)
    vec_env = model.get_env()
    if vec_env is None:
        print("Venv is none")
        exit(1)
    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)


if __name__ == "__main__":
    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    main()
