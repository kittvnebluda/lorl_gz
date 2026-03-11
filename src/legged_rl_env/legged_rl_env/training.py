from stable_baselines3 import PPO, SAC

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
