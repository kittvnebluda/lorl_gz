from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    PathSubstitution,
)
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ros_gz_bridge.actions import RosGzBridge


def generate_launch_description():
    # Package shared directories
    legged_rl_gazebo_pkg_share = FindPackageShare("legged_rl_gazebo")
    legged_rl_bringup_pkg_share = FindPackageShare("legged_rl_bringup")
    legged_rl_description_pkg_share = FindPackageShare("legged_rl_description")
    ros_gz_sim_pkg_share = FindPackageShare("ros_gz_sim")
    unitree_go2_sim_pkg_share = FindPackageShare("unitree_go2_sim")

    # Set default arguments
    default_robot_description = PathJoinSubstitution(
        [legged_rl_description_pkg_share, "urdf", "go2.urdf.xacro"]
    )
    default_gazebo_world = PathJoinSubstitution(
        [legged_rl_gazebo_pkg_share, "worlds", "empty.sdf"]
    )
    default_rviz_cfg = PathJoinSubstitution(
        [legged_rl_bringup_pkg_share, "rviz", "go2.rviz"]
    )

    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument("world", default_value=default_gazebo_world)
    )
    declared_arguments.append(
        DeclareLaunchArgument("debug", default_value="false", choices=["true", "false"])
    )

    # Initialize arguments
    use_sim_time = True
    joints_config = PathJoinSubstitution(
        [unitree_go2_sim_pkg_share, "config", "joints", "joints.yaml"]
    )
    gait_config = PathJoinSubstitution(
        [unitree_go2_sim_pkg_share, "config", "gait", "gait.yaml"]
    )
    links_config = PathJoinSubstitution(
        [unitree_go2_sim_pkg_share, "config", "links", "links.yaml"]
    )
    robot_controllers = PathJoinSubstitution(
        [legged_rl_bringup_pkg_share, "config", "go2_controllers.yaml"]
    )
    robot_urdf = Command(
        [
            "xacro ",
            default_robot_description,
            " robot_controllers:=",
            robot_controllers,
            " DEBUG:=",
            LaunchConfiguration("debug"),
        ]
    )
    ros_gz_bridge_cfg = PathJoinSubstitution(
        [legged_rl_bringup_pkg_share, "config", "ros_gz_bridge.yaml"]
    )

    # Set common parameters
    parameters = [SetParameter(name="use_sim_time", value=use_sim_time)]

    # Declare nodes
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_urdf}],
    )
    gazebo_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([ros_gz_sim_pkg_share, "launch", "gz_sim.launch.py"])
        ),
        launch_arguments={
            "gz_args": [PathSubstitution(LaunchConfiguration("world")), " -r"]
        }.items(),
    )
    gazebo_spawn_entity_node = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "/robot_description"],
    )
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )
    delay_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=gazebo_spawn_entity_node,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )
    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "effort_controller",
            "--param-file",
            robot_controllers,
        ],
    )
    delay_robot_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[robot_controller_spawner],
        )
    )
    gazebo_bridge = RosGzBridge(
        bridge_name="ros_gazebo_bridge", config_file=ros_gz_bridge_cfg
    )
    foot_contacts_node = Node(
        package="legged_rl_applications",
        executable="foot_contact_aggregator",
        output="both",
    )
    state_estimator_node = Node(
        package="champ_base",
        executable="state_estimation_node",
        output="screen",
        parameters=[
            {"orientation_from_imu": True},
            {"urdf": robot_urdf},
            joints_config,
            links_config,
            gait_config,
        ],
        remappings=[("/imu/data", "/imu")],
    )
    base_to_footprint_ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="base_to_footprint_ekf",
        output="screen",
        parameters=[
            {"base_link_frame": "base_link"},
            PathJoinSubstitution(
                [
                    FindPackageShare("champ_base"),
                    "config",
                    "ekf",
                    "base_to_footprint.yaml",
                ]
            ),
        ],
        remappings=[("/imu/data", "/imu")],
    )
    footprint_to_odom_ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="footprint_to_odom_ekf",
        output="screen",
        parameters=[
            {"base_link_frame": "base_footprint"},
            PathJoinSubstitution(
                [
                    FindPackageShare("champ_base"),
                    "config",
                    "ekf",
                    "footprint_to_odom.yaml",
                ]
            ),
        ],
        remappings=[("/imu/data", "/imu"), ("/odom/filtered", "/odom/raw")],
    )
    map_to_odom_tf_node = Node(
        package="tf2_ros",
        name="map_to_odom_tf_node",
        executable="static_transform_publisher",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            "map",
            "--child-frame-id",
            "odom",
        ],
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", default_rviz_cfg],
    )

    nodes = [
        robot_state_publisher_node,
        gazebo_sim_launch,
        gazebo_bridge,
        gazebo_spawn_entity_node,
        delay_joint_state_broadcaster_spawner,
        delay_robot_controller_spawner,
        foot_contacts_node,
        state_estimator_node,
        base_to_footprint_ekf_node,
        footprint_to_odom_ekf_node,
        map_to_odom_tf_node,
        rviz_node,
    ]

    return LaunchDescription(declared_arguments + parameters + nodes)
