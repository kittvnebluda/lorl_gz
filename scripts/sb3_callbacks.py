import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


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
    def __init__(self, params, rtf, random_seed, verbose: int = 0):
        super().__init__(verbose)
        self.params = params
        self.rtf = rtf
        self.random_seed = random_seed

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "model_seed": int(self.random_seed),
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
