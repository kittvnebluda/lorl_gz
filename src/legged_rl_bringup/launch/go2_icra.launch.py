import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    legged_rl_gazebo_pkg_path = get_package_share_directory("legged_rl_gazebo")
    legged_rl_description_pkg_path = get_package_share_directory(
        "legged_rl_description"
    )
    unitree_go2_sim_pkg_path = get_package_share_directory("unitree_go2_sim")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map", default_value="flat", choices=["flat", "sloped"]
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        unitree_go2_sim_pkg_path, "launch", "unitree_go2_launch.py"
                    )
                ),
                launch_arguments={
                    "world": os.path.join(
                        legged_rl_gazebo_pkg_path, "worlds", "empty.sdf"
                    ),
                    "world_init_x": "6",
                    "world_init_y": "-3",
                    "world_init_z": "1",
                    "world_init_heading": "3.14",
                    "unitree_go2_description_path": os.path.join(
                        legged_rl_description_pkg_path, "urdf", "go2.urdf.xacro"
                    ),
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
