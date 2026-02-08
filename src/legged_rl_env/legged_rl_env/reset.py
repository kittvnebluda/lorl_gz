import tempfile
import logging
import time
from subprocess import CompletedProcess, run
import rclpy

from rclpy.node import Node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gz_srv_cmd(
    service: str, reqtype: str, reptype: str, timeout: int, request: str
) -> CompletedProcess:
    cmd = [
        "gz",
        "service",
        "-s",
        str(service),
        "--reqtype",
        str(reqtype),
        "--reptype",
        str(reptype),
        "--timeout",
        str(timeout),
        "--req",
        str(request),
    ]
    return run(cmd, capture_output=True)


def gz_srv_rm(world_name: str, entity_type: int, entity_name: str):
    res = gz_srv_cmd(
        service=f"/world/{world_name}/remove",
        reqtype="gz.msgs.Entity",
        reptype="gz.msgs.Boolean",
        timeout=2000,
        request=f"type: {entity_type}, name: '{entity_name}'",
    )
    if res.returncode == 0:
        logger.info(f"Removed entity {entity_name} successful")
    else:
        logger.info(f"Error: {res.stderr}")


def create_entity_from_robot_description(world_name: str, entity_name: str, z: float):
    cmd = [
        "ros2",
        "run",
        "ros_gz_sim",
        "create",
        "--world",
        str(world_name),
        "--topic",
        "robot_description",
        "--name",
        str(entity_name),
        "--z",
        str(z),
    ]
    res = run(cmd, capture_output=True)
    if res.returncode == 0:
        logger.info(f"Created entity {entity_name} successful")
    else:
        logger.info(f"Error: {res.stderr}")


class CmdGazeboEnvResetter:
    @classmethod
    def reset(cls, world_name: str, entity_name: str):
        gz_srv_rm(world_name, 2, entity_name)
        create_entity_from_robot_description(world_name, entity_name, 0.4)
