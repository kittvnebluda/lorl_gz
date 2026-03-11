from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
from geometry_msgs.msg import Quaternion
from rclpy.executors import Executor
from rclpy.task import Future
from tf_transformations import quaternion_from_euler


@dataclass
class ServiceCall:
    name: str
    future: Future
    handler: Optional[Callable]


def bool_res_handler(name: str, res: bool, logger):
    if not res:
        logger.error(f"{name} failed")
        raise RuntimeError(f"{name} returned failure")


def wait_for_all_futures(
    executor: Executor,
    futures: Iterable[ServiceCall],
    logger,
    timeout_sec: float = 5.0,
) -> None:
    for it in futures:
        name, future, res_handler = it.name, it.future, it.handler
        executor.spin_until_future_complete(future, timeout_sec=timeout_sec)
        while not future.done():
            logger.warn(f"{name} timed out, waiting again...")
            executor.spin_until_future_complete(future, timeout_sec=timeout_sec)
        if res_handler is not None:
            res_handler(name, future.result(), logger)


def norm11(x, low, high):
    """Normalize to [-1; 1]"""
    return np.clip(2.0 * (x - low) / (high - low) - 1.0, -1, 1)


def denorm11(x_norm, low, high):
    """Denormalize from [-1; 1]"""
    return low + (x_norm + 1.0) * 0.5 * (high - low)


def norm01(x, low, high):
    """Normalize to [0; 1]"""
    return np.clip((x - low) / (high - low), 0, 1)


def denorm01(x_norm, low, high):
    """Denormalize from [0; 1]"""
    return x_norm * (high - low) + low


def quat_msg_from_euler(x, y, z):
    x, y, z, w = quaternion_from_euler(x, y, z)
    return Quaternion(x=x, y=y, z=z, w=w)
