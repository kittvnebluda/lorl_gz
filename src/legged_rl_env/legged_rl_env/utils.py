from dataclasses import dataclass
from typing import Callable, Optional

from rclpy.executors import Executor
from rclpy.task import Future


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
    futures: list[ServiceCall],
    logger,
    timeout_sec: float = 5.0,
) -> None:
    for it in futures:
        name, future, res_handler = it.name, it.future, it.handler
        executor.spin_until_future_complete(future, timeout_sec=timeout_sec)
        if not future.done():
            logger.error(f"{name} timeout")
            raise TimeoutError(f"{name} execution timed out")

        if res_handler is not None:
            res_handler(name, future.result(), logger)
