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
            buffer_size=1_000_000,
            learning_starts=100,
            batch_size=1024,
            tau=0.005,
            gamma=0.99,
            train_freq=2,
            gradient_steps=1,
            ent_coef="auto",
            target_entropy="auto",
        ),
    },
}
