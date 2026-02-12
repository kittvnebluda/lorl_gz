import numpy as np
from gymnasium import ObservationWrapper, spaces


class FlattenObsDict(ObservationWrapper):
    """
    Flattens Dict observation spaces into a single Box.

    Works recursively and supports nested Dict spaces,
    as long as leaves are Box spaces.
    """

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("Observation space must be spaces.Dict")

        self._keys, low, high = self._flatten_space(env.observation_space)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def _flatten_space(self, space, prefix=""):
        keys = []
        lows = []
        highs = []

        for key, subspace in space.spaces.items():
            full_key = f"{prefix}{key}"

            if isinstance(subspace, spaces.Dict):
                sub_keys, sub_low, sub_high = self._flatten_space(
                    subspace, prefix=full_key + "."
                )
                keys.extend(sub_keys)
                lows.append(sub_low)
                highs.append(sub_high)

            elif isinstance(subspace, spaces.Box):
                keys.append(full_key)
                lows.append(subspace.low.flatten())
                highs.append(subspace.high.flatten())

            else:
                raise TypeError(
                    f"Unsupported subspace type: {type(subspace)} for key {key}"
                )

        return keys, np.concatenate(lows), np.concatenate(highs)

    def observation(self, observation):
        flat_parts = []
        self._flatten_obs(observation, flat_parts)
        return np.concatenate(flat_parts).astype(np.float32)

    def _flatten_obs(self, obs, parts):
        for key in self.env.observation_space.spaces.keys():  # pyright: ignore[reportAttributeAccessIssue]
            value = obs[key]

            if isinstance(value, dict):
                self._flatten_obs(value, parts)
            else:
                parts.append(np.asarray(value).flatten())
