import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    ros_gz_sim_pkg_path = get_package_share_directory("ros_gz_sim")
    legged_rl_description_pkg_path = get_package_share_directory(
        "legged_rl_description"
    )
    legged_rl_gazebo_pkg_path = get_package_share_directory("legged_rl_gazebo")

    gz_launch_path = os.path.join(ros_gz_sim_pkg_path, "launch", "gz_sim.launch.py")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map", default_value="flat", choices=["flat", "sloped"]
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(gz_launch_path),
                launch_arguments={
                    "gz_args": [
                        os.path.join(legged_rl_gazebo_pkg_path, "worlds", "empty.sdf")
                    ],
                    "on_exit_shutdown": "True",
                }.items(),
            ),
            Node(
                package="ros_gz_sim",
                executable="create",
                parameters=[
                    {
                        "name": "map",
                        "file": PathJoinSubstitution(
                            [
                                legged_rl_description_pkg_path,
                                "urdf",
                                ["icra_map_", LaunchConfiguration("map"), ".urdf"],
                            ]
                        ),
                        "use_sim_time": True,
                    }
                ],
            ),
        ]
    )
